#include "OpenCLUtils.h"
#include <iostream>

std::vector<cl::Platform> OpenCLUtils::get_Platfoms() {
	std::vector<cl::Platform> all_Platform;
	cl::Platform::get(&all_Platform);
	return all_Platform;
}

std::vector<cl::Device> OpenCLUtils::get_Devices_for_Platform(cl::Platform platform, cl_device_type device_type) {
	std::vector<cl::Device> all_devices;
	platform.getDevices(device_type, &all_devices);
	return all_devices;
}

cl::Device OpenCLUtils::get_Default_Device() {
	return get_Devices_for_Platform(get_Platfoms()[0], CL_DEVICE_TYPE_ALL)[0];
}

std::string  OpenCLUtils::get_Kernel_Code(const char* filename) {
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.reserve(in.tellg());
		in.seekg(0, std::ios::beg);
		contents.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		in.close();
		return(contents);
	}
	throw(errno);
}

cl::Program OpenCLUtils::build_Program_from_Source(cl::Device device, cl::Context context, const char* filename) {
	// create CL Program from kernel code
	cl::Program::Sources sources;
	std::string src_code = get_Kernel_Code(filename);

	// add kernel code string to sources
	sources.push_back({ src_code.c_str(), src_code.length() });

	// create Program
	cl::Program program(context, sources);

	// build Program
	cl_int build_res = program.build({ device });

	// check for errors
	if (build_res != CL_SUCCESS) {
		//std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		throw(errno);
	}

	return program;
}