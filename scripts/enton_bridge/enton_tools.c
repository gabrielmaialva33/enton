/*
 * enton_tools — multicall binary with basic utilities missing from the
 * Yoosee SC-B21 busybox (1.30.1 compiled without tail/head/strings/xxd/file).
 *
 * Build:
 *   zig cc -target arm-linux-musleabi -mcpu=arm926ej_s -static -Os \
 *       -o enton_tools enton_tools.c
 *
 * Deploy: drop on /mnt/disc1/enton_tools on the camera and create
 *   symlinks /tmp/bin/{tail,head,strings,xxd,hd,file}. The tool dispatches
 *   via argv[0] basename.
 *
 * Supports: tail, head, strings, xxd, hd (alias for xxd -g1), file (basic)
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* -- TAIL ------------------------------------------------------------- */

static int tool_tail(int argc, char **argv) {
    int n = 10;
    const char *fname = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == 'n') {
            n = atoi(a[2] ? a + 2 : argv[++i]);
        } else if (a[0] == '-' && isdigit((unsigned char)a[1])) {
            n = atoi(a + 1);
        } else {
            fname = a;
        }
    }
    if (n < 0) n = -n;
    if (n == 0) n = 1;
    FILE *f = fname ? fopen(fname, "r") : stdin;
    if (!f) { perror(fname ? fname : "stdin"); return 1; }

    char **buf = calloc((size_t)n, sizeof(char *));
    size_t *len = calloc((size_t)n, sizeof(size_t));
    if (!buf || !len) { fprintf(stderr, "oom\n"); return 2; }

    size_t cap = 8192;
    char *line = malloc(cap);
    int count = 0;
    for (;;) {
        ssize_t nread = getline(&line, &cap, f);
        if (nread <= 0) break;
        int slot = count % n;
        free(buf[slot]);
        buf[slot] = strndup(line, (size_t)nread);
        len[slot] = (size_t)nread;
        count++;
    }
    int start = count < n ? 0 : count - n;
    for (int i = start; i < count; i++) {
        int slot = i % n;
        if (buf[slot]) fwrite(buf[slot], 1, len[slot], stdout);
    }
    for (int i = 0; i < n; i++) free(buf[i]);
    free(buf); free(len); free(line);
    if (fname) fclose(f);
    return 0;
}

/* -- HEAD ------------------------------------------------------------- */

static int tool_head(int argc, char **argv) {
    int n = 10;
    const char *fname = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == 'n') {
            n = atoi(a[2] ? a + 2 : argv[++i]);
        } else if (a[0] == '-' && isdigit((unsigned char)a[1])) {
            n = atoi(a + 1);
        } else {
            fname = a;
        }
    }
    if (n < 0) n = -n;
    FILE *f = fname ? fopen(fname, "r") : stdin;
    if (!f) { perror(fname ? fname : "stdin"); return 1; }
    size_t cap = 8192; char *line = malloc(cap);
    int count = 0;
    while (count < n) {
        ssize_t nread = getline(&line, &cap, f);
        if (nread <= 0) break;
        fwrite(line, 1, (size_t)nread, stdout);
        count++;
    }
    free(line);
    if (fname) fclose(f);
    return 0;
}

/* -- STRINGS ---------------------------------------------------------- */

static int tool_strings(int argc, char **argv) {
    int min_len = 4;
    const char *fname = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] == '-' && a[1] == 'n') {
            min_len = atoi(a[2] ? a + 2 : argv[++i]);
        } else {
            fname = a;
        }
    }
    FILE *f = fname ? fopen(fname, "rb") : stdin;
    if (!f) { perror(fname ? fname : "stdin"); return 1; }
    char *buf = malloc((size_t)min_len + 1024);
    int used = 0, cap = min_len + 1024;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c >= 0x20 && c < 0x7f) {
            if (used >= cap - 1) { cap *= 2; buf = realloc(buf, (size_t)cap); }
            buf[used++] = (char)c;
        } else {
            if (used >= min_len) { buf[used] = 0; puts(buf); }
            used = 0;
        }
    }
    if (used >= min_len) { buf[used] = 0; puts(buf); }
    free(buf);
    if (fname) fclose(f);
    return 0;
}

/* -- XXD / HD --------------------------------------------------------- */

static int tool_xxd(int argc, char **argv, int group1) {
    int groupsize = group1 ? 1 : 2;  /* hd default = -g1 */
    int cols = 16;
    const char *fname = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-g") && i + 1 < argc) groupsize = atoi(argv[++i]);
        else if (!strncmp(a, "-g", 2)) groupsize = atoi(a + 2);
        else if (!strcmp(a, "-c") && i + 1 < argc) cols = atoi(argv[++i]);
        else if (!strncmp(a, "-c", 2)) cols = atoi(a + 2);
        else fname = a;
    }
    if (groupsize < 1) groupsize = 1;
    if (cols < 1) cols = 16;

    FILE *f = fname ? fopen(fname, "rb") : stdin;
    if (!f) { perror(fname ? fname : "stdin"); return 1; }
    uint8_t *buf = malloc((size_t)cols);
    size_t off = 0;
    size_t n;
    while ((n = fread(buf, 1, (size_t)cols, f)) > 0) {
        printf("%08zx: ", off);
        for (size_t i = 0; i < (size_t)cols; i++) {
            if (i < n) printf("%02x", buf[i]); else printf("  ");
            if ((i + 1) % groupsize == 0) putchar(' ');
        }
        putchar(' ');
        for (size_t i = 0; i < n; i++) putchar(buf[i] >= 0x20 && buf[i] < 0x7f ? buf[i] : '.');
        putchar('\n');
        off += n;
    }
    free(buf);
    if (fname) fclose(f);
    return 0;
}

/* -- FILE (basic) ----------------------------------------------------- */

static int tool_file(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: file <path>\n"); return 2; }
    for (int i = 1; i < argc; i++) {
        const char *p = argv[i];
        struct stat st;
        if (lstat(p, &st) < 0) { printf("%s: cannot stat: %s\n", p, strerror(errno)); continue; }
        if (S_ISDIR(st.st_mode)) { printf("%s: directory\n", p); continue; }
        if (S_ISLNK(st.st_mode)) {
            char tgt[512];
            ssize_t r = readlink(p, tgt, sizeof(tgt) - 1);
            if (r > 0) { tgt[r] = 0; printf("%s: symbolic link to %s\n", p, tgt); continue; }
        }
        if (!S_ISREG(st.st_mode)) { printf("%s: special file\n", p); continue; }

        int fd = open(p, O_RDONLY);
        if (fd < 0) { printf("%s: %s\n", p, strerror(errno)); continue; }
        uint8_t h[32] = {0};
        ssize_t r = read(fd, h, sizeof(h));
        close(fd);
        if (r < 4) { printf("%s: empty or short\n", p); continue; }

        if (h[0] == 0x7f && h[1] == 'E' && h[2] == 'L' && h[3] == 'F') {
            const char *cls = (h[4] == 1) ? "32-bit" : (h[4] == 2) ? "64-bit" : "?";
            const char *end = (h[5] == 1) ? "LSB" : (h[5] == 2) ? "MSB" : "?";
            int machine = h[18] | (h[19] << 8);
            const char *arch = "?";
            if (machine == 3) arch = "x86";
            else if (machine == 40) arch = "ARM";
            else if (machine == 62) arch = "x86_64";
            else if (machine == 183) arch = "AArch64";
            else if (machine == 8) arch = "MIPS";
            int type = h[16] | (h[17] << 8);
            const char *typestr = (type == 1) ? "relocatable" : (type == 2) ? "executable" : (type == 3) ? "shared object" : "?";
            printf("%s: ELF %s %s %s %s\n", p, cls, end, arch, typestr);
            continue;
        }
        if (h[0] == '#' && h[1] == '!') { printf("%s: script (%.20s...)\n", p, h + 2); continue; }
        if (h[0] == 0x1f && h[1] == 0x8b) { printf("%s: gzip\n", p); continue; }
        if (h[0] == 0xfd && h[1] == '7' && h[2] == 'z' && h[3] == 'X') { printf("%s: xz\n", p); continue; }
        if (h[0] == 0x89 && h[1] == 'P' && h[2] == 'N' && h[3] == 'G') { printf("%s: PNG\n", p); continue; }
        if (h[0] == 0xff && h[1] == 0xd8) { printf("%s: JPEG\n", p); continue; }
        if (!memcmp(h, "hsqs", 4)) { printf("%s: SquashFS LE\n", p); continue; }
        if (!memcmp(h, "qshs", 4)) { printf("%s: SquashFS BE\n", p); continue; }

        /* try text vs binary */
        int printable = 0;
        for (int i = 0; i < r; i++) if ((h[i] >= 0x20 && h[i] < 0x7f) || h[i] == '\n' || h[i] == '\t') printable++;
        if (printable * 10 / r >= 9) printf("%s: ASCII text\n", p);
        else printf("%s: data (%02x %02x %02x %02x ...)\n", p, h[0], h[1], h[2], h[3]);
    }
    return 0;
}

/* -- dispatch --------------------------------------------------------- */

static const char *basename_of(const char *p) {
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

int main(int argc, char **argv) {
    const char *me = basename_of(argv[0]);
    /* If first arg looks like an applet name, allow busybox-style call:
     *   enton_tools tail -5 foo
     */
    if (!strcmp(me, "enton_tools") && argc >= 2) {
        me = argv[1];
        argv++; argc--;
    }
    if (!strcmp(me, "tail"))    return tool_tail(argc, argv);
    if (!strcmp(me, "head"))    return tool_head(argc, argv);
    if (!strcmp(me, "strings")) return tool_strings(argc, argv);
    if (!strcmp(me, "xxd"))     return tool_xxd(argc, argv, 0);
    if (!strcmp(me, "hd"))      return tool_xxd(argc, argv, 1);
    if (!strcmp(me, "file"))    return tool_file(argc, argv);

    fprintf(stderr,
        "enton_tools — multicall binary. Applets: tail, head, strings, xxd, hd, file\n"
        "Usage:\n"
        "  enton_tools <applet> [args...]\n"
        "  or symlink as /tmp/bin/<applet> and call directly.\n"
        "\nInvoked as: '%s'\n", me);
    return 2;
}
