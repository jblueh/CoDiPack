#include "../build/allTests.hpp"
#include "../include/testInterface.hpp"
#include "../include/tests/allTests.hpp"

template<typename Dummy>
void listNames(TestNames& names) {
  (void)names;
}

template<typename Dummy, typename Test, typename... Args>
void listNames(TestNames& names) {
  Test test{};
  names.insert(test.getName());

  listNames<Dummy, Args...>(names);
}

void listAllNames(TestNames& names) {
  listNames<void, ALL_TESTS>(names);
}
