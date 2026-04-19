# Enton Robot Body — Yoosee SC-B21

O corpo do Enton é uma câmera IP **Yoosee SC-B21** dual-lens PTZ com firmware modificado pra expor root shell + ONVIF control. Hardware hack completo feito em 18/Abr/2026.

## Hardware Specs

| Componente       | Spec                                                        |
|------------------|-------------------------------------------------------------|
| **SoC**          | Anyka **AK3918AV100** (ARM926EJ-S ARMv5TEJ, 441 BogoMIPS)   |
| **Kernel**       | Linux 4.4.282 (build May/2024)                              |
| **Userland**     | Buildroot 2018.02.7, uClibc-ng 1.0.31, BusyBox 1.30.1       |
| **Flash**        | SPI NOR 8MB (W25Q64 compatible)                             |
| **RAM**          | 32MB (devtmpfs=16MB)                                        |
| **Firmware**     | Gwell/Yoosee 38.2.103 (git `825e0295`, build 2026-03-02)    |
| **Sensors**      | Dual: Sony SC2331 MIPI + outro (SC1346/SC2336 disponíveis)  |
| **Encoder**      | HW H.264/H.265, 8 canais (`ak_venc` driver)                 |
| **Audio**        | Anyka `ak_pcm` (mic + speaker, G.711 8kHz)                  |
| **Motor**        | 2-axis stepper (pan+tilt), driver `presetmotor` com hrtimer |
| **WiFi**         | SSV6x5x + BLE (Bilian A8:B5:8E OUI)                         |
| **Ethernet**     | Gwelltimes 4C:B0:08 OUI                                     |
| **USB**          | OTG host (suporta modems 3G/4G + RNDIS ethernet)            |
| **Storage ext.** | microSD slot (fat32, mmcblk0p1)                             |
| **ADC**          | `ak_saradc` 12-bit (sensor de luz + botões analógicos)      |

## Device Identity

- **Device ID** (DevSerial Yoosee): `5893329321`
- **MAC eth0**: `4C:B0:08:8A:BD:EC`
- **MAC wlan0**: `A8:B5:8E:BF:8B:53`
- **IP default** (lab): `192.168.1.5` (DHCP)
- **Hardware Ver**: 2.1

## Access Channels

### 1. Telnet Root Shell (backdoor persistente)

```bash
telnet 192.168.1.5 23    # ou 2323
```

- Shell root **sem login** (`telnetd -l /bin/sh`)
- Persistente via modificação em `/etc/init.d/rc.local` na flash (mtd5)
- **Known limit**: busybox minimal, sem `head`, `pidof`, `strace`, `python`, `gdb`

### 2. ONVIF SOAP — Motor PTZ + Controle

- **Endpoint**: `http://192.168.1.5:5000/onvif/{device,media,ptz}_service`
- **Auth**: nenhuma (zero credentials needed)
- **ProfileToken**: `IPCProfilesToken0` (descoberto via `GetProfiles` do `media_service`)
  - **NÃO usar `PROFILE_000`** — é token fantasma, retorna 200 OK mas não move o motor
  - `IPCProfilesToken1` também existe e aponta pro SubStream
- **Calls que FUNCIONAM**: `GetProfiles`, `ContinuousMove`, `GetSystemDateAndTime`, `GetCapabilities`
- **Calls que TRAVAM O IPC** (evitar sempre): `Stop`, `GetStatus`, `CreateUsers`, `SetHostname`, operações write
- Pra parar o motor: mandar `ContinuousMove` com `x=0 y=0` (NÃO chamar `Stop`)
- **⚠️ DoS-sensitive**: PTZ floods sem throttle ≥500ms crasham IPC. Requer power cycle pra recovery.

### 3. RTSP Stream

- **Dual-lens sem auth**: `rtsp://192.168.1.5:554/cam/realmonitor` (1920x2160 H.265 stitched vertical)
- **Stream 2** (requer auth): `rtsp://USER:PASS@192.168.1.5:554/onvif1`
- **Sub-stream**: `/onvif2` (640x720)

### 4. Serial UART (backup physical)

- Pinos UART na PCB: ttySAK0 a **115200 8N1** (activo no inittab)
- Precisa USB-TTL (3.3V logic) + desmontar o case
- Login prompt aparece mas precisa crackear hash MD5-crypt

## Filesystem Map

```
/                   squashfs RO  (mtd5 ROOTFS 1.15MB — contém nosso backdoor)
/ipc                squashfs RO  (mtd6 APP 4.77MB — Gwell application overlay)
/rom                jffs2 RW     (mtd7 ROM 256KB — config persistente)
/etc                tmpfs bind   (copiado de / no boot, perdeu no reboot)
/tmp, /var, /mnt    tmpfs
/mnt/ramdisk        tmpfs
```

## Key Devices (`/dev/`)

| Device | Purpose |
|---|---|
| `/dev/preset_motor` | Motor PTZ via ioctl (char 10,58) — **usar ONVIF em vez disso** |
| `/dev/pcmC0D0p` | ALSA speaker playback |
| `/dev/pcmC0D0c` | ALSA mic capture |
| `/dev/pcmC0D0l` | ALSA loopback |
| `/dev/video-0-{0,1,2}` | Câmera 1 (main/sub/snap streams) |
| `/dev/video-1-{0,1,2}` | Câmera 2 (dual-lens) |
| `/dev/venc-chn0..7` | HW encoder 8 channels |
| `/dev/isp-param-{0,1}` | ISP config |
| `/dev/isp-stats-{ae,af,awb,3dnr}-{0,1}` | ISP statistics |
| `/dev/v4l-subdev{0,1}` | V4L2 subdevs |
| `/dev/iio:device0` | IIO (light ADC provável) |
| `/dev/watchdog` | Watchdog timer |
| `/dev/ttySAK{0,1,2}` | 3 UARTs (0=console, 1+2 livres) |

## GPIOs Ativos

| Pin | Direction | Default Value | Função (inferida) |
|---|---|---|---|
| `gpio19` | out | 1 | LED status / IR LED enable |
| `gpio37` | out | 1 | ? |
| `gpio48` | out | 0 | Motor direction? |
| `gpio56` | in | 1 | **Botão reset** (pressed=0) |
| `gpio67` | out | 0 | ? |
| `gpio68` | out | 0 | ? |
| `gpio98` | out | 1 | ? |

Toggle via telnet: `echo 1 > /sys/class/gpio/gpioXX/value`

## MTD Partitions (SPI Flash 8MB)

```
mtd0   UBOOT     0x037000  bootloader
mtd1   ENV       0x001000  u-boot env
mtd2   ENVBK     0x001000  env backup
mtd3   DTB       0x010000  device tree
mtd4   KERNEL    0x180000  Linux kernel (XZ compressed)
mtd5   ROOTFS    0x120000  base rootfs (/ → squashfs) ← backdoor lives here
mtd6   APP       0x4D7000  /ipc app overlay (Gwell IPC)
mtd7   ROM       0x040000  /rom config (JFFS2 RW)
mtd8   FLASH     0x800000  full flash mapping
```

## Secrets vazadas (encontradas no JFFS2)

- **WiFi SSID/PSK**: `DENIS` / `12345678` (plaintext em `wpa_supplicant0.conf`)
- **Huawei Cloud OBS**: 5 pares AKS/SKS em `ivStorage/iv_cs_local_info.bin`
- **Tenant**: `dophigo2019-huaweicloud`
- **RSA priv key**: `/etc/keyrsa/2307km01n.prv` (formato Anyka IASRSA, ~1024-bit)
- **User MD5 hash**: `764ec3743569d159547cae02a3f31dfa` (rockyou + numeric 6-8 + IoT defaults: sem match — é custom pw)

## Known Issues / Watch Out

1. **ONVIF DoS** — flood de requests (<500ms interval) trava o IPC main process. Requer power cycle.
1a. **ONVIF `Stop`/`GetStatus` = reboot garantido** — handler desses verbos tá bugado no firmware Gwell 38.2.103. Pra parar motor, usar `ContinuousMove` com `x=0 y=0`. Pra ler posição, via telnet `dmesg | grep H_now_pos | awk 'END{print}'`.
2. **Block watchdog** — câmera reboota em 60s se watchdog não for feeded por app
3. **Reboot perde /etc** — é tmpfs bind. Modificações runtime em `/etc/*` não persistem. Só modificações via flash (mtd5) são permanentes.
4. **`/rom` (mtd7 JFFS2) é writable** — bom pra persistir config sem precisar reflashar
5. **Flash write via SPI é lento** — ~10min com ESP32-serprog a 4Mbaud+spispeed=8M
6. **Single-core ARM926EJ-S 441MHz** — processing leve, deixa vision+LLM no host

## Hack History

Ver [firmware_hack.md](./firmware_hack.md) pro procedimento completo de extração SPI + patch + flash.

## Reference Docs

- OpenIPC AK3918EV200 ISP reverse: https://github.com/OpenIPC/ak3918ev200
- OpenIPC motors: https://github.com/OpenIPC/motors
- Anyka_ak3918 hacking journey: https://github.com/VGerris/Anyka_ak3918_hacking_journey
- IOT-ANYKA-PTZdaemon: https://github.com/kuhnchris/IOT-ANYKA-PTZdaemon
- Repo principal pentest lab: `~/Documents/pentest/camera-hack/`
