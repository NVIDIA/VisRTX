#pragma once

#include <mi/base/ilogger.h>
#include <mi/base/interface_implement.h>

namespace visrtx {
struct DeviceGlobalState;
} // namespace visrtx

namespace visrtx::mdl {

class Logger : public mi::base::Interface_implement<mi::base::ILogger>
{
 public:
  Logger(DeviceGlobalState *deviceState) : m_deviceState(deviceState) {}

  void message(mi::base::Message_severity level,
      const char *module_category,
      const mi::base::Message_details &,
      const char *message) override;
  void message(mi::base::Message_severity level,
      const char *module_category,
      const char *message) override;

 private:
  DeviceGlobalState *m_deviceState;
};

} // namespace visrtx::mdl
