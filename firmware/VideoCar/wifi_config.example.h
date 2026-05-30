/*
  WiFi configuration for the ESP32-CAM Video Car.

  HOW TO USE:
    cp wifi_config.example.h wifi_config.h
  then edit wifi_config.h with your real network.
  wifi_config.h is git-ignored so your password never lands in the repo.

  Two modes (set `ap`):
    ap = false -> STA: car joins your router. IP is printed over serial @115200.
                  NOTE: ESP32 only sees 2.4 GHz networks, NOT 5 GHz.
    ap = true  -> AP : car creates its own hotspot, reachable at 192.168.4.1.
*/
#pragma once

// --- mode ---
bool ap = false;  // false = join router (STA), true = own hotspot (AP)

// --- credentials ---
const char* ssid     = "YOUR_2.4GHZ_SSID";  // router SSID (STA) or hotspot name (AP)
const char* password = "YOUR_PASSWORD";     // network password (>=8 chars for AP; blank = open)

// --- AP-only settings (ignored in STA mode) ---
int channel       = 11;  // WiFi channel for AP mode
int hidden        = 0;   // 1 = hidden SSID
int maxconnection = 1;   // max simultaneous clients in AP mode
