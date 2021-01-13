/*print a message by multi threads*/
#include <cstdint> //uint64_t
#include <vector> //std::vector
#include <thread> //std::thread
#include <iostream> //std::cout

//std::threads() constructor receive any number of parameter
//the first parameter is for the function(it must return void), the others are for the function's parameter
//in this example, function is say_hello, and it's parameter is uint64_t id

//this function will be called by the threads
void say_hello(uint64_t id) {
	std::cout << "hello from thread: " << id << std::endl;
}

//main runs in the master thread
int main(int argc, char *argv[]) {
	
	//create threads according to command line parameter
	if (argc != 2) {
		std::cout << "usage: hello_world.x num_threads";
		return -1;
	}
	const uint64_t num_threads = atoi(argv[1]);

	//create a vector to store the thread
	std::vector<std::thread> threads;
	//or you can just store it in the array using bellow command
	//std::thread* threads = new std::thread[num_threads];
	
	//create threads and make it run the function say_hello()
	for (uint64_t id = 0; id < num_threads; id++)
		//threads.emplace_back(say_hello, id);
		threads.push_back(std::thread(say_hello, id)); //use this because it is easy to understand

	//join all the threads
	for (auto& thread: threads)
		thread.join();

	//delete the array if you create it dynamically
	//delete [] threads

	return 0;
}
