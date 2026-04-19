/*
 * enton_bridge — minimal TCP daemon for the Enton physical body
 * (Yoosee SC-B21 / Anyka AK3918AV100, Linux 4.4.282 uClibc-ng).
 *
 * Runs on the camera alongside /ipc/ipc; exposes a single TCP socket
 * on port 9999 that the host brain talks to instead of juggling ONVIF
 * SOAP + telnet + RTSP + direct ioctl.
 *
 * Wire protocol: line-based ASCII. One command per line, CR/LF tolerated.
 *   Request:  CMD arg1 arg2 ... \n
 *   Response: OK <data> \n     or     ERR <msg> \n
 *
 * Supported commands (v0.2):
 *   PING                          -> "OK pong"
 *   GPIO_SET <num> <0|1>          -> set GPIO value via /sys/class/gpio
 *   GPIO_GET <num>                -> read GPIO value
 *   MOTOR_RAW <H> <V>             -> ioctl cmd=3 on /dev/preset_motor
 *                                    (only works if /ipc/ipc not holding it)
 *   PAN <dir> [iters=10] [speed=0.5] -> burst of ContinuousMove requests
 *                                    dir: left|right|up|down; iters 1-100;
 *                                    speed 0.0-1.0. Firmware needs a burst;
 *                                    a single request is ignored silently.
 *   HALT                          -> ContinuousMove x=0 y=0 (safe stop)
 *   POS                           -> last H_now_pos,V_now_pos from dmesg
 *   EXEC <shell_cmd...>           -> runs via sh -c, returns up to 4KB stdout
 *   DMESG_TAIL <n>                -> last n dmesg lines
 *   STAT                          -> uptime, load, free mem
 *   BYE                           -> close this connection
 *
 * Build (host):
 *   zig cc -target arm-linux-musleabi -mcpu=arm926ej_s -static -Os \
 *       -o enton_bridge enton_bridge.c
 *
 * Deploy (from host, camera reachable via telnet on $CAM_IP):
 *   ncat -l -p 9997 --send-only < enton_bridge &
 *   telnet $CAM_IP 23
 *     > nc -w 3 $HOST_IP 9997 > /mnt/disc1/enton_bridge
 *     > chmod +x /mnt/disc1/enton_bridge
 *     > /mnt/disc1/enton_bridge &
 *
 * Autostart: patched /etc/init.d/rc.local adds the last line.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PORT            9999
#define BACKLOG         4
#define LINE_MAX_LEN    4096
#define REPLY_MAX_LEN   16384

/* -- motor ioctl layout (from presetmotor.ko RE, cmd indices discovered 2026-04-18) */
#define MOTOR_DEV       "/dev/preset_motor"
#define MOTOR_CMD_RUN_TO_POS 3  /* dwMotorRunToPos(H, V) */
#define MOTOR_CMD_GET_POS    4  /* copy_to_user: returns struct (often zero) */
#define MOTOR_BUF_SIZE  40      /* kernel buffer, 10 int32 */

/* -- ONVIF loopback targets */
#define ONVIF_HOST      "127.0.0.1"
#define ONVIF_PORT      5000
#define ONVIF_PROFILE   "IPCProfilesToken0"
#define PTZ_ACTION      "http://www.onvif.org/ver20/ptz/wsdl/ContinuousMove"

/* -- helpers ---------------------------------------------------------- */

static void logf(const char *fmt, ...) {
    char ts[32];
    time_t t = time(NULL);
    strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&t));
    fprintf(stderr, "[bridge %s] ", ts);
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
}

static int write_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        ssize_t r = write(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) return -1;
        p += r; n -= (size_t)r;
    }
    return 0;
}

static void reply(int fd, const char *fmt, ...) {
    char buf[REPLY_MAX_LEN];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf) - 2, fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    if (n >= (int)sizeof(buf) - 2) n = sizeof(buf) - 2;
    buf[n++] = '\n';
    (void)write_all(fd, buf, (size_t)n);
}

/* Read a file entirely into a heap buffer; returns NULL on error. */
static char *slurp(const char *path, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { close(fd); return NULL; }
    for (;;) {
        if (len + 1024 >= cap) { cap *= 2; char *nb = realloc(buf, cap); if (!nb) { free(buf); close(fd); return NULL; } buf = nb; }
        ssize_t r = read(fd, buf + len, cap - len - 1);
        if (r < 0) { if (errno == EINTR) continue; free(buf); close(fd); return NULL; }
        if (r == 0) break;
        len += (size_t)r;
    }
    buf[len] = '\0';
    close(fd);
    if (out_len) *out_len = len;
    return buf;
}

/* -- command handlers ------------------------------------------------- */

static void cmd_ping(int fd) { reply(fd, "OK pong"); }

static void cmd_gpio_set(int fd, const char *num_s, const char *val_s) {
    int gpio = atoi(num_s);
    int val = atoi(val_s);
    if (val != 0 && val != 1) { reply(fd, "ERR val must be 0 or 1"); return; }

    char path[64];
    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/value", gpio);
    int g = open(path, O_WRONLY);
    if (g < 0) { reply(fd, "ERR open %s: %s", path, strerror(errno)); return; }
    const char *v = val ? "1" : "0";
    if (write(g, v, 1) != 1) { close(g); reply(fd, "ERR write: %s", strerror(errno)); return; }
    close(g);
    reply(fd, "OK gpio%d=%d", gpio, val);
}

static void cmd_gpio_get(int fd, const char *num_s) {
    int gpio = atoi(num_s);
    char path[64];
    snprintf(path, sizeof(path), "/sys/class/gpio/gpio%d/value", gpio);
    size_t n;
    char *c = slurp(path, &n);
    if (!c) { reply(fd, "ERR open %s: %s", path, strerror(errno)); return; }
    int val = (c[0] == '1') ? 1 : 0;
    free(c);
    reply(fd, "OK gpio%d=%d", gpio, val);
}

/*
 * MOTOR — absolute target via cmd=3 (dwMotorRunToPos).
 * The driver's kernel state machine handles stepper timing; the app just
 * deposits the target (H, V) and the hw_timer moves the motor gradually.
 */
static void cmd_motor(int fd, const char *h_s, const char *v_s) {
    int32_t target_h = atoi(h_s);
    int32_t target_v = atoi(v_s);
    int m = open(MOTOR_DEV, O_RDWR);
    if (m < 0) { reply(fd, "ERR open %s: %s", MOTOR_DEV, strerror(errno)); return; }

    uint8_t buf[MOTOR_BUF_SIZE];
    memset(buf, 0, sizeof(buf));
    /* cmd=3 layout from RE: args consumed at [0x00] and [0x04] as int32 */
    buf[0x00] = (uint8_t)(target_h & 0xff);
    buf[0x01] = (uint8_t)((target_h >> 8) & 0xff);
    buf[0x02] = (uint8_t)((target_h >> 16) & 0xff);
    buf[0x03] = (uint8_t)((target_h >> 24) & 0xff);
    buf[0x04] = (uint8_t)(target_v & 0xff);
    buf[0x05] = (uint8_t)((target_v >> 8) & 0xff);
    buf[0x06] = (uint8_t)((target_v >> 16) & 0xff);
    buf[0x07] = (uint8_t)((target_v >> 24) & 0xff);

    if (ioctl(m, MOTOR_CMD_RUN_TO_POS, buf) < 0) {
        int err = errno;
        close(m);
        reply(fd, "ERR ioctl: %s", strerror(err));
        return;
    }
    close(m);
    reply(fd, "OK motor target=%d,%d", target_h, target_v);
}

/*
 * Send an HTTP POST to 127.0.0.1:5000 with a SOAP body and a SOAPAction header.
 * Waits for the response status line; does not parse the body.
 * Returns 0 on success (any 2xx), -1 on error.
 */
static int onvif_post(const char *endpoint_path, const char *soap_action, const char *body) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ONVIF_PORT);
    inet_aton(ONVIF_HOST, &addr.sin_addr);

    struct timeval tv = {.tv_sec = 3, .tv_usec = 0};
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) { close(sock); return -1; }

    char req[4096];
    int n = snprintf(req, sizeof(req),
        "POST %s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: application/soap+xml;charset=UTF-8;action=\"%s\"\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        endpoint_path, ONVIF_HOST, ONVIF_PORT, soap_action, strlen(body), body);
    if (n < 0 || n >= (int)sizeof(req)) { close(sock); return -1; }

    if (write_all(sock, req, (size_t)n) < 0) { close(sock); return -1; }

    /* Read status line */
    char resp[256];
    ssize_t r = read(sock, resp, sizeof(resp) - 1);
    close(sock);
    if (r <= 0) return -1;
    resp[r] = '\0';
    /* Expect "HTTP/1.1 2xx ..." */
    if (strncmp(resp, "HTTP/1.", 7) != 0) return -1;
    int code = atoi(resp + 9);
    return (code >= 200 && code < 300) ? 0 : -1;
}

/* Send ContinuousMove with given PanTilt velocities. Returns 0 on OK. */
static int onvif_continuous_move(double x, double y) {
    char body[1024];
    snprintf(body, sizeof(body),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<s:Envelope xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\">"
        "<s:Body><ContinuousMove xmlns=\"http://www.onvif.org/ver20/ptz/wsdl\">"
        "<ProfileToken>%s</ProfileToken>"
        "<Velocity><PanTilt x=\"%.2f\" y=\"%.2f\" "
        "xmlns:tt=\"http://www.onvif.org/ver10/schema\"/></Velocity>"
        "</ContinuousMove></s:Body></s:Envelope>",
        ONVIF_PROFILE, x, y);
    return onvif_post("/onvif/ptz_service", PTZ_ACTION, body);
}

/*
 * PAN <dir> [iters=10] [speed=0.5]
 * Firmware needs *sustained* ContinuousMove bursts to actually move the motor;
 * a single request is accepted (200 OK) but ignored by the state machine.
 * Empirically: 20 requests @ 500ms ≈ 32 step movement.
 */
static void cmd_pan(int fd, const char *dir, const char *iters_s, const char *spd_s) {
    int iters = iters_s ? atoi(iters_s) : 10;
    if (iters < 1) iters = 1;
    if (iters > 100) iters = 100;
    double speed = spd_s ? atof(spd_s) : 0.5;
    if (speed < 0.0) speed = 0.0;
    if (speed > 1.0) speed = 1.0;

    double x = 0, y = 0;
    if      (!strcasecmp(dir, "left"))  x = -speed;
    else if (!strcasecmp(dir, "right")) x =  speed;
    else if (!strcasecmp(dir, "up"))    y =  speed;
    else if (!strcasecmp(dir, "down"))  y = -speed;
    else { reply(fd, "ERR direction must be left|right|up|down"); return; }

    int ok = 0, fail = 0;
    for (int i = 0; i < iters; i++) {
        if (onvif_continuous_move(x, y) == 0) ok++; else fail++;
        /* 500ms between requests (firmware throttle window) */
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 500 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    reply(fd, "OK pan %s iters=%d ok=%d fail=%d", dir, iters, ok, fail);
}

static void cmd_halt(int fd) {
    /* ContinuousMove x=0 y=0: halts motor without calling ONVIF Stop (which crashes IPC). */
    if (onvif_continuous_move(0.0, 0.0) < 0) { reply(fd, "ERR halt failed"); return; }
    reply(fd, "OK halted");
}

static void cmd_pos(int fd) {
    /* Parse dmesg for the last H_now_pos / V_now_pos line. */
    FILE *p = popen("dmesg 2>/dev/null | awk '/H_now_pos/{h=$0} END{print h}'", "r");
    if (!p) { reply(fd, "ERR popen: %s", strerror(errno)); return; }
    char buf[256]; buf[0] = '\0';
    if (fgets(buf, sizeof(buf), p) == NULL) buf[0] = '\0';
    pclose(p);
    size_t n = strlen(buf);
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
    if (n == 0) { reply(fd, "OK nopos"); return; }

    /* Extract just "H=80 V=30" from the full line. */
    int h = -1, v = -1;
    const char *hp = strstr(buf, "H_now_pos = ");
    const char *vp = strstr(buf, "V_now_pos = ");
    if (hp) h = atoi(hp + 12);
    if (vp) v = atoi(vp + 12);
    reply(fd, "OK H=%d V=%d", h, v);
}

static void cmd_exec(int fd, const char *shell_line) {
    if (!shell_line || !*shell_line) { reply(fd, "ERR empty"); return; }
    FILE *p = popen(shell_line, "r");
    if (!p) { reply(fd, "ERR popen: %s", strerror(errno)); return; }

    char buf[REPLY_MAX_LEN - 64];
    size_t total = 0;
    for (;;) {
        size_t r = fread(buf + total, 1, sizeof(buf) - 1 - total, p);
        if (r == 0) break;
        total += r;
        if (total >= sizeof(buf) - 1) break;
    }
    pclose(p);
    buf[total] = '\0';
    /* Strip trailing whitespace and escape newlines for single-line reply */
    while (total > 0 && (buf[total-1] == '\n' || buf[total-1] == '\r' || buf[total-1] == ' ')) buf[--total] = '\0';
    /* Replace newlines with \\n (pipe-safe single-line reply) */
    for (size_t i = 0; i < total; i++) if (buf[i] == '\n') buf[i] = '|';
    reply(fd, "OK %s", buf);
}

static void cmd_dmesg_tail(int fd, const char *n_s) {
    int n = n_s ? atoi(n_s) : 20;
    if (n < 1) n = 1;
    if (n > 200) n = 200;
    char cmd_buf[160];
    snprintf(cmd_buf, sizeof(cmd_buf),
             "dmesg | awk 'END{for(i=NR-%d+1;i<=NR;i++)print a[i%%%d]} {a[NR%%%d]=$0}'",
             n, n, n);
    cmd_exec(fd, cmd_buf);
}

static void cmd_stat(int fd) {
    size_t n;
    char *up = slurp("/proc/uptime", &n);
    char *ld = slurp("/proc/loadavg", &n);
    char *mi = slurp("/proc/meminfo", &n);
    char out[1024]; int off = 0;
    off += snprintf(out + off, sizeof(out) - off, "uptime=%s", up ? up : "?");
    off += snprintf(out + off, sizeof(out) - off, " load=%s", ld ? ld : "?");
    /* MemFree line */
    if (mi) {
        char *mf = strstr(mi, "MemFree:");
        if (mf) {
            char *eol = strchr(mf, '\n'); if (eol) *eol = '\0';
            off += snprintf(out + off, sizeof(out) - off, " %s", mf);
        }
    }
    /* strip newlines */
    for (int i = 0; i < off; i++) if (out[i] == '\n' || out[i] == '\r') out[i] = ' ';
    free(up); free(ld); free(mi);
    reply(fd, "OK %s", out);
}

/* -- line dispatch ---------------------------------------------------- */

static int tokenize(char *line, char **argv, int max) {
    int argc = 0;
    char *p = line;
    while (*p && argc < max) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        argv[argc++] = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        if (*p) *p++ = '\0';
    }
    return argc;
}

static int handle_line(int fd, char *line) {
    /* trim */
    while (*line == ' ' || *line == '\t') line++;
    size_t n = strlen(line);
    while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r' || line[n-1] == ' ')) line[--n] = '\0';
    if (n == 0) return 0;

    char *argv[16];
    /* For EXEC we want the rest of the line intact. */
    if (!strncasecmp(line, "EXEC ", 5)) { cmd_exec(fd, line + 5); return 0; }

    int argc = tokenize(line, argv, 16);
    if (argc == 0) return 0;

    if (!strcasecmp(argv[0], "PING"))         { cmd_ping(fd); return 0; }
    if (!strcasecmp(argv[0], "BYE"))          { reply(fd, "OK bye"); return 1; }
    if (!strcasecmp(argv[0], "STAT"))         { cmd_stat(fd); return 0; }
    if (!strcasecmp(argv[0], "POS"))          { cmd_pos(fd); return 0; }
    if (!strcasecmp(argv[0], "HALT"))         { cmd_halt(fd); return 0; }
    if (!strcasecmp(argv[0], "GPIO_SET") && argc == 3) { cmd_gpio_set(fd, argv[1], argv[2]); return 0; }
    if (!strcasecmp(argv[0], "GPIO_GET") && argc == 2) { cmd_gpio_get(fd, argv[1]); return 0; }
    if (!strcasecmp(argv[0], "MOTOR_RAW") && argc == 3) { cmd_motor(fd, argv[1], argv[2]); return 0; }
    if (!strcasecmp(argv[0], "PAN") && argc >= 2 && argc <= 4) {
        cmd_pan(fd, argv[1],
                argc >= 3 ? argv[2] : NULL,
                argc >= 4 ? argv[3] : NULL);
        return 0;
    }
    if (!strcasecmp(argv[0], "DMESG_TAIL") && argc == 2) { cmd_dmesg_tail(fd, argv[1]); return 0; }

    reply(fd, "ERR unknown: %s", argv[0]);
    return 0;
}

static void serve_client(int cfd) {
    char buf[LINE_MAX_LEN];
    size_t fill = 0;
    reply(cfd, "OK enton_bridge v0.1 ready");
    for (;;) {
        ssize_t r = read(cfd, buf + fill, sizeof(buf) - 1 - fill);
        if (r < 0) { if (errno == EINTR) continue; break; }
        if (r == 0) break;
        fill += (size_t)r;
        buf[fill] = '\0';
        /* process one line at a time */
        char *nl;
        while ((nl = memchr(buf, '\n', fill)) != NULL) {
            *nl = '\0';
            int done = handle_line(cfd, buf);
            size_t consumed = (size_t)(nl - buf) + 1;
            memmove(buf, buf + consumed, fill - consumed);
            fill -= consumed;
            if (done) return;
        }
        if (fill >= sizeof(buf) - 1) { reply(cfd, "ERR line too long"); return; }
    }
}

/* Create /tmp/bin symlinks for enton_tools applets so they're on PATH. */
static void bootstrap_tools(void) {
    const char *src = "/mnt/disc1/enton_tools";
    struct stat st;
    if (stat(src, &st) != 0) return;  /* silently skip if tools not present */
    mkdir("/tmp/bin", 0755);
    const char *applets[] = { "tail", "head", "strings", "xxd", "hd", "file", NULL };
    for (const char **a = applets; *a; a++) {
        char dst[64];
        snprintf(dst, sizeof(dst), "/tmp/bin/%s", *a);
        unlink(dst);  /* ignore error */
        if (symlink(src, dst) == 0) logf("symlinked /tmp/bin/%s -> %s", *a, src);
    }
}

int main(int argc, char **argv) {
    int port = PORT;
    if (argc > 1) port = atoi(argv[1]);

    signal(SIGPIPE, SIG_IGN);
    signal(SIGCHLD, SIG_IGN);  /* auto-reap */

    bootstrap_tools();

    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { logf("socket: %s", strerror(errno)); return 1; }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) { logf("bind :%d: %s", port, strerror(errno)); return 1; }
    if (listen(s, BACKLOG) < 0) { logf("listen: %s", strerror(errno)); return 1; }

    logf("listening on :%d", port);

    for (;;) {
        struct sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        int cfd = accept(s, (struct sockaddr *)&peer, &plen);
        if (cfd < 0) { if (errno == EINTR) continue; logf("accept: %s", strerror(errno)); continue; }

        pid_t pid = fork();
        if (pid == 0) {
            close(s);
            logf("client %s:%d", inet_ntoa(peer.sin_addr), ntohs(peer.sin_port));
            serve_client(cfd);
            close(cfd);
            _exit(0);
        } else if (pid > 0) {
            close(cfd);
        } else {
            logf("fork: %s", strerror(errno));
            close(cfd);
        }
    }
    /* unreachable */
    return 0;
}
