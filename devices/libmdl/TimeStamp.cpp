#include "TimeStamp.h"

namespace visrtx::libmdl {

TimeStamp newTimeStamp()
{
  static TimeStamp ts = 0;

  return ++ts;
}

} // namespace visrtx::libmdl