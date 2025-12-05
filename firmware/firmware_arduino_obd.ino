/*
  firmware_arduino_obd.ino
  - Arduino UNO + MCP2515 (CS D10, INT D2) @ 500 kbps
  - Envia requests OBD-II (modo 01) para PIDs: 0x0C RPM, 0x0D Speed, 0x05 Coolant, 0x11 TPS, 0x10 MAF, 0x0B MAP
  - Recebe 0x7E8/0x7E9 e imprime CSV: rpm,speed,coolant_temp,tps,maf,map @115200
  Lib necessária: "mcp_can" (Cory J. Fowler)
*/
#include <SPI.h>
#include "mcp_can.h"

#define CAN_CS 10
#define CAN_INT 2

MCP_CAN CAN0(CAN_CS);

unsigned long lastReqMs = 0;
const unsigned long reqIntervalMs = 100; // ~10 Hz total (ciclo de PIDs)

// PIDs da rodada
const byte pidList[] = {0x0C, 0x0D, 0x05, 0x11, 0x10, 0x0B};
const int pidCount = sizeof(pidList)/sizeof(pidList[0]);
int pidIndex = 0;

// cache de valores
int rpm = -1;
int speed = -1;
int coolant = -100;
int tps = -1;
int maf = -1; // g/s *100? manter cru (A*256+B)/100
int mapv = -1;

void sendOBDRequest(byte mode, byte pid) {
  // ID funcional 0x7DF, 8 bytes
  byte data[8] = {0x02, mode, pid, 0x55, 0x55, 0x55, 0x55, 0x55};
  CAN0.sendMsgBuf(0x7DF, 0, 8, data);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  pinMode(CAN_INT, INPUT);

  if (CAN0.begin(MCP_ANY, CAN_500KBPS, MCP_8MHZ) == CAN_OK) {
    Serial.println(F("# CAN init OK"));
  } else {
    Serial.println(F("# CAN init FAIL"));
    while (1);
  }
  CAN0.setMode(MCP_NORMAL);
  delay(500);
}

void requestNextPID() {
  sendOBDRequest(0x01, pidList[pidIndex]);
  pidIndex = (pidIndex + 1) % pidCount;
}

void loop() {
  unsigned long now = millis();

  // disparo cíclico de requisições
  if (now - lastReqMs >= reqIntervalMs) {
    lastReqMs = now;
    requestNextPID();
  }

  // leitura de respostas
  if (!digitalRead(CAN_INT)) {
    long unsigned int rxId;
    unsigned char len = 0;
    unsigned char buf[8];
    if (CAN0.readMsgBuf(&rxId, &len, buf) == CAN_OK) {
      // esperamos 0x7E8..0x7EF
      if ((rxId & 0x7F0) == 0x7E8 && len >= 3) {
        // resposta: [len, 0x41, PID, A, B, ...]
        if (buf[1] == 0x41) {
          byte pid = buf[2];
          switch (pid) {
            case 0x0C: { // RPM
              int A = buf[3];
              int B = buf[4];
              rpm = ((A * 256) + B) / 4;
            } break;
            case 0x0D: { // Speed
              int A = buf[3];
              speed = A;
            } break;
            case 0x05: { // Coolant
              int A = buf[3];
              coolant = A - 40;
            } break;
            case 0x11: { // TPS
              int A = buf[3];
              tps = (int)( (A * 100.0) / 255.0 + 0.5 );
            } break;
            case 0x10: { // MAF
              int A = buf[3];
              int B = buf[4];
              maf = (A * 256 + B); // valor bruto: /100 = g/s
            } break;
            case 0x0B: { // MAP
              int A = buf[3];
              mapv = A; // kPa
            } break;
          }
          // imprime uma linha quando completamos um ciclo (heurística simples)
          if (pid == 0x0B) {
            // CSV: rpm,speed,coolant_temp,tps,maf,map
            // maf sai em unidades brutas (divide por 100 no Python se quiser g/s)
            Serial.print(rpm); Serial.print(",");
            Serial.print(speed); Serial.print(",");
            Serial.print(coolant); Serial.print(",");
            Serial.print(tps); Serial.print(",");
            Serial.print(maf); Serial.print(",");
            Serial.println(mapv);
          }
        }
      }
    }
  }
}
