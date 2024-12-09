#include "Logger.h"

#include "optix_visrtx.h"

#include <anari/frontend/anari_enums.h>

#include <fmt/format.h>

namespace {

ANARIStatusSeverity miSeverityToAnari(mi::base::Message_severity severity)
{
  switch (severity) {
  case mi::base::MESSAGE_SEVERITY_ERROR:
    return ANARI_SEVERITY_ERROR;
  case mi::base::MESSAGE_SEVERITY_WARNING:
    return ANARI_SEVERITY_WARNING;
  case mi::base::MESSAGE_SEVERITY_INFO:
    return ANARI_SEVERITY_INFO;
  case mi::base::MESSAGE_SEVERITY_VERBOSE:
    return ANARI_SEVERITY_INFO;
  case mi::base::MESSAGE_SEVERITY_DEBUG:
    return ANARI_SEVERITY_DEBUG;
  default:
    return ANARI_SEVERITY_INFO;
  }
}

} // namespace

namespace visrtx::mdl {

void Logger::message(mi::base::Message_severity level,
    const char *moduleCategory,
    const mi::base::Message_details &details,
    const char *message)
{
  this->message(level, moduleCategory, message);
}

void Logger::message(mi::base::Message_severity level,
    const char *moduleCategory,
    const char *message)
{
  m_deviceState->messageFunction(miSeverityToAnari(level),
      fmt::format("[VISRTX:MDL:{}] {}", moduleCategory, message),
      ANARI_UNKNOWN,
      nullptr);
}

} // namespace visrtx::mdl
