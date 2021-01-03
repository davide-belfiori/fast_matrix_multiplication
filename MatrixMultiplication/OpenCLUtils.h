#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <streambuf>
#include <CL/cl.hpp>

class OpenCLUtils
{
public:
	static std::vector<cl::Platform> get_Platfoms();
	static std::vector<cl::Device> get_Devices_for_Platform(cl::Platform platform, cl_device_type device_type);
	static cl::Device get_Default_Device();
	std::string get_Kernel_Code(const char* filename);
	cl::Program build_Program_from_Source(cl::Device device, cl::Context context, const char* filename);
};

