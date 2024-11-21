#include <algorithm>
#include <fstream>
#include <perfcpp/hardware_info.h>
#include <sstream>

std::optional<std::uint64_t>
perf::HardwareInfo::intel_pebs_mem_loads_aux_event_id()
{
  if (HardwareInfo::is_intel()) {
    return HardwareInfo::parse_event_umask_from_file("/sys/bus/event_source/devices/cpu/events/mem-loads-aux");
  }

  return std::nullopt;
}

std::optional<std::uint64_t>
perf::HardwareInfo::intel_pebs_mem_loads_event_id()
{
  if (HardwareInfo::is_intel()) {
    return HardwareInfo::parse_event_umask_from_file("/sys/bus/event_source/devices/cpu/events/mem-loads");
  }

  return std::nullopt;
}

std::optional<std::uint64_t>
perf::HardwareInfo::intel_pebs_mem_stores_event_id()
{
  if (HardwareInfo::is_intel()) {
    return HardwareInfo::parse_event_umask_from_file("/sys/bus/event_source/devices/cpu/events/mem-stores");
  }

  return std::nullopt;
}

std::optional<std::uint64_t>
perf::HardwareInfo::parse_event_umask_from_file(std::string&& path)
{
  auto event_stream = std::ifstream{ path };
  if (event_stream.is_open()) {
    std::string line;
    std::getline(event_stream, line);

    if (!line.empty()) {
      /// The line should look like "event=0xcd,umask=0x1,ldlat=3".

      auto event = std::optional<std::string>{ std::nullopt };
      auto umask = std::optional<std::string>{ std::nullopt };

      auto token_stream = std::stringstream{line};
      std::string token;

      /// Process every token where tokens are separated by ','.
      while (std::getline(token_stream, token, ',')) {

        /// Locate eq-char.
        const auto pos = token.find('=');
        if (pos == std::string::npos) {
          continue;
        }

        auto key = token.substr(0ULL, pos);
        auto value = token.substr(pos + 1ULL);

        /// Remove possible whitespace
        key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());

        /// Convert key to lowercase for case-insensitivity
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);

        /// Remove the values "0x" prefix.
        if (value.rfind("0x", 0ULL) == 0ULL) {
          value = value.substr(2ULL);
        }

        if (key == "event") {
          event = std::move(value);
        } else if (key == "umask") {
          umask = std::move(value);
        }
      }

      /// Combine event and umask to a single event id.
      if (event.has_value() && umask.has_value()) {
        return std::stoull(/* combine <umask><event> */ umask.value().append(event.value()), nullptr, 16);
      }
    }
  }

  return std::nullopt;
}

std::optional<std::uint32_t>
perf::HardwareInfo::amd_ibs_op_type()
{
  if (perf::HardwareInfo::is_amd_ibs_supported()) {
    auto ibs_op_stream = std::ifstream{ "/sys/bus/event_source/devices/ibs_op/type" };
    if (ibs_op_stream.is_open()) {
      std::uint32_t type;
      ibs_op_stream >> type;

      return type;
    }
  }

  return std::nullopt;
}

std::optional<std::uint32_t>
perf::HardwareInfo::amd_ibs_fetch_type()
{
  if (perf::HardwareInfo::is_amd_ibs_supported()) {
    auto ibs_op_stream = std::ifstream{ "/sys/bus/event_source/devices/ibs_fetch/type" };
    if (ibs_op_stream.is_open()) {
      std::uint32_t type;
      ibs_op_stream >> type;

      return type;
    }
  }

  return std::nullopt;
}