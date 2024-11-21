#include <cerrno>
#include <iostream>
#include <perfcpp/group.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>

using namespace perf;

bool
perf::Group::open(const perf::Config config)
{
  /// File descriptor of the group leader.
  auto group_leader_file_descriptor = -1LL;

  for (auto counter_id = 0U; counter_id < this->_members.size(); ++counter_id) {
    auto& counter = this->_members[counter_id];
    const auto is_group_leader = counter_id == 0U;

    counter.open(config.is_debug(),
                 is_group_leader,
                 /* is_secret_leader */ false,
                 group_leader_file_descriptor,
                 config.cpu_id(),
                 config.process_id(),
                 config.is_include_child_threads(),
                 config.is_include_kernel(),
                 config.is_include_user(),
                 config.is_include_hypervisor(),
                 config.is_include_idle(),
                 config.is_include_guest(),
                 /* is_read_format */ true,
                 /* sample_type */ std::nullopt,
                 /* branch_type */ std::nullopt,
                 /* user_registers */ std::nullopt,
                 /* kernel_registers */ std::nullopt,
                 /* max_callstack */ std::nullopt,
                 /* is_include_context_switch */ false,
                 /* is_include_cgroup */ false);

    /// Set the group leader file descriptor.
    if (is_group_leader) {
      group_leader_file_descriptor = counter.file_descriptor();
    }
  }

  /// If we cannot open any counter, we will throw an exception.
  return true;
}

void
perf::Group::close()
{
  for (auto& counter : this->_members) {
    counter.close();
  }
}

bool
perf::Group::start()
{
  if (this->_members.empty()) {
    throw std::runtime_error{ "Cannot start an empty group." };
  }

  const auto leader_file_descriptor = static_cast<std::int32_t>(this->leader_file_descriptor());

  ::ioctl(leader_file_descriptor, PERF_EVENT_IOC_RESET, 0);
  ::ioctl(leader_file_descriptor, PERF_EVENT_IOC_ENABLE, 0);

  const auto read_size =
    ::read(leader_file_descriptor, &this->_start_value, sizeof(CounterReadFormat<Group::MAX_MEMBERS>));
  return read_size > 0ULL;
}

bool
perf::Group::stop()
{
  if (this->_members.empty()) {
    return false;
  }

  const auto leader_file_descriptor = static_cast<std::int32_t>(this->leader_file_descriptor());

  const auto read_size =
    ::read(leader_file_descriptor, &this->_end_value, sizeof(CounterReadFormat<Group::MAX_MEMBERS>));
  ::ioctl(leader_file_descriptor, PERF_EVENT_IOC_DISABLE, 0);
  return read_size > 0ULL;
}

bool
perf::Group::add(perf::CounterConfig counter)
{
  this->_members.emplace_back(counter);
  return true;
}

double
perf::Group::get(const std::size_t index) const
{
  const auto multiplexing_correction = double(this->_end_value.time_enabled - this->_start_value.time_enabled) /
                                       double(this->_end_value.time_running - this->_start_value.time_running);

  const auto& counter = this->_members[index];
  const auto start_value = Group::value_for_id(this->_start_value, counter.id());
  const auto end_value = Group::value_for_id(this->_end_value, counter.id());

  if (start_value.has_value() && end_value.has_value()) {
    const auto result = double(end_value.value() - start_value.value());
    return result > .0 ? result * multiplexing_correction : .0;
  }

  return 0;
}

std::optional<std::uint64_t>
perf::Group::value_for_id(const CounterReadFormat<Group::MAX_MEMBERS>& counter_values, std::uint64_t id) noexcept
{
  for (auto i = 0U; i < counter_values.count_members; ++i) {
    if (counter_values.values[i].id == id) {
      return counter_values.values[i].value;
    }
  }

  return std::nullopt;
}
