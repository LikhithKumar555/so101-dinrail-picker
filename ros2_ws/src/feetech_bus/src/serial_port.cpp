#include "feetech_bus/serial_port.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/file.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace feetech
{

namespace
{
speed_t to_speed(int baud)
{
  switch (baud) {
    case 1000000: return B1000000;
    case 500000: return B500000;
    case 115200: return B115200;
    case 57600: return B57600;
    default: return B0;
  }
}
}  // namespace

SerialPort::~SerialPort() {close();}

bool SerialPort::open(const std::string & device, int baud, std::string * error)
{
  close();
  const speed_t speed = to_speed(baud);
  if (speed == B0) {
    if (error) {*error = "unsupported baud rate " + std::to_string(baud);}
    return false;
  }
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
  if (fd_ < 0) {
    if (error) {*error = device + ": " + std::strerror(errno);}
    return false;
  }
  // One program per bus. Two programs reading the same port steal each other's replies: both
  // see random timeouts and neither knows why. flock() takes an exclusive lock on the port; a
  // second program gets EWOULDBLOCK at once. The kernel drops the lock when this fd is closed -
  // also when the program crashes - so a crash never leaves the port locked.
  if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
    if (error) {
      *error = device + ": " + (errno == EWOULDBLOCK ?
        std::string("in use by another program (one driver per bus)") : std::strerror(errno));
    }
    close();
    return false;
  }
  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) {
    if (error) {*error = std::string("tcgetattr: ") + std::strerror(errno);}
    close();
    return false;
  }
  cfmakeraw(&tio);                       // no echo, no line editing, no CR/LF translation, 8 bits
  tio.c_cflag |= CLOCAL | CREAD;         // ignore modem lines, enable the receiver
  tio.c_cflag &= ~(CSTOPB | PARENB);     // 1 stop bit, no parity
  tio.c_cc[VMIN] = 0;                    // read() never blocks by itself - poll() does the waiting
  tio.c_cc[VTIME] = 0;
  cfsetispeed(&tio, speed);
  cfsetospeed(&tio, speed);
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    if (error) {*error = std::string("tcsetattr: ") + std::strerror(errno);}
    close();
    return false;
  }
  tcflush(fd_, TCIOFLUSH);               // drop bytes left over from an earlier session
  return true;
}

void SerialPort::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialPort::write_all(const uint8_t * data, std::size_t n)
{
  std::size_t done = 0;
  while (done < n) {
    const ssize_t w = ::write(fd_, data + done, n - done);
    if (w < 0) {
      if (errno == EINTR || errno == EAGAIN) {continue;}
      return false;
    }
    done += static_cast<std::size_t>(w);
  }
  return true;
}

ssize_t SerialPort::read_some(uint8_t * buf, std::size_t capacity, std::chrono::microseconds timeout)
{
  pollfd p{fd_, POLLIN, 0};
  const int ms = static_cast<int>((timeout.count() + 999) / 1000);   // round up to whole ms
  const int r = ::poll(&p, 1, ms);
  if (r < 0) {
    return errno == EINTR ? 0 : -1;
  }
  if (r == 0) {
    return 0;
  }
  if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) {
    return -1;
  }
  const ssize_t n = ::read(fd_, buf, capacity);
  if (n < 0) {
    return (errno == EINTR || errno == EAGAIN) ? 0 : -1;
  }
  return n;
}

void SerialPort::discard_input()
{
  tcflush(fd_, TCIFLUSH);
}

}  // namespace feetech
