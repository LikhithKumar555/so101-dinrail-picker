// POSIX serial port in raw mode (termios), with reads that wait at most a given time (poll).
//
// Raw mode matters for a binary protocol: in the default "cooked" mode the terminal layer
// edits the byte stream (echo, CR -> LF, Ctrl-C = signal, Ctrl-S = flow control) and would
// change or swallow bytes such as 0x0D, 0x03 or 0x13.
#pragma once

#include <sys/types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace feetech
{

class SerialPort
{
public:
  SerialPort() = default;
  ~SerialPort();
  SerialPort(const SerialPort &) = delete;
  SerialPort & operator=(const SerialPort &) = delete;

  // open in raw 8N1 mode at the given baud rate (1000000, 500000, 115200, 57600)
  bool open(const std::string & device, int baud, std::string * error = nullptr);
  void close();
  bool is_open() const {return fd_ >= 0;}

  bool write_all(const uint8_t * data, std::size_t n);
  // bytes read (> 0), 0 if nothing arrived within timeout, -1 on an I/O error
  ssize_t read_some(uint8_t * buf, std::size_t capacity, std::chrono::microseconds timeout);
  // throw away everything that is already waiting in the input buffer
  void discard_input();

private:
  int fd_ = -1;
};

}  // namespace feetech
