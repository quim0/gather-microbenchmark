#pragma once

#include "config.h"
#include "counter.h"

namespace perf {
/**
 * A group presents a set of counters where the first counter is the group leader.
 * All counters can be started and stopped together, not individually.
 */
class Group
{
public:
  /// Number of maximal members per group.
  constexpr static inline auto MAX_MEMBERS = 8U;

  Group() = default;
  Group(Group&&) noexcept = default;
  Group(const Group&) = default;

  ~Group() = default;

  /**
   * Adds the given event to the group.
   *
   * @param counter Event to add.
   * @return True, if the event could be added.
   */
  bool add(CounterConfig counter);

  /**
   * Opens all counters of the group, configured by the provided config.
   *
   * @param config Configuration.
   *
   * @return True, if the counters could be opened.
   */
  bool open(Config config);

  /**
   * Closes all counters of the group.
   */
  void close();

  /**
   * Starts monitoring the counters in the group.
   *
   * @return True, if the counters could be started.
   */
  bool start();

  /**
   * Stops monitoring of all counters in the group.
   *
   * @return True, if the counters could be stopped.
   */
  bool stop();

  /**
   * @return Number of counters in the group.
   */
  [[nodiscard]] std::size_t size() const noexcept { return _members.size(); }

  /**
   * @return True, if the group is empty.
   */
  [[nodiscard]] bool empty() const noexcept { return _members.empty(); }

  /**
   * @return The file descriptor of the group leader (or -1 if the group is empty).
   */
  [[nodiscard]] std::int64_t leader_file_descriptor() const noexcept
  {
    return !_members.empty() ? _members.front().file_descriptor() : -1LL;
  }

  /**
   * Reads the result of counter at the given index.
   *
   * @param index Index of the counter to read the result for.
   * @return Result of the counter.
   */
  [[nodiscard]] double get(std::size_t index) const;

  /**
   * Grants access to the counter at the given index.
   *
   * @param index Index of the counter.
   * @return Counter.
   */
  [[nodiscard]] Counter& member(const std::size_t index) { return _members[index]; }

  /**
   * Grants access to the counter at the given index.
   *
   * @param index Index of the counter.
   * @return Counter.
   */
  [[nodiscard]] const Counter& member(const std::size_t index) const { return _members[index]; }

  /**
   * @return List of all members in the group.
   */
  [[nodiscard]] std::vector<Counter>& members() { return _members; }

private:
  /// List of all the group members.
  std::vector<Counter> _members;

  /// Start value of the hardware performance counters.
  CounterReadFormat<Group::MAX_MEMBERS> _start_value{};

  /// End value of the hardware performance counters.
  CounterReadFormat<Group::MAX_MEMBERS> _end_value{};

  /**
   * Reads the value of a specific counter (identified by the given ID) from the provided value set.
   *
   * @param counter_values Set of counter values.
   * @param id Identifier of the counter to read.
   *
   * @return The value of the specified counter or std::nullopt of the ID was not found.
   */
  [[nodiscard]] static std::optional<std::uint64_t> value_for_id(
    const CounterReadFormat<Group::MAX_MEMBERS>& counter_values,
    std::uint64_t id) noexcept;
};
}