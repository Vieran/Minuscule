#include <cstdint>
#include <vector>
#include <thread>
#include <iostream>

void say_hello(uint64_t id) {
	std::cout << "hello from thread: " << id << std::endl;
}

//main runs in the master thread
int main(int argc, char *argv[]) {

	const uint64_t num_threads = 4;
	std::vector<std::thread> threads;
	
	//for all threads
	for (uint64_t id = 0; id < num_threads; id++)
		threads.emplace_back(say_hello, id);

	for (auto& thread: threads)
		thread.join();

	return 0;
}
