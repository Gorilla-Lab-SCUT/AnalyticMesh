
#include <torch/extension.h>
#include <ostream>
#include <iomanip>
#include <sstream>

#include "utilities.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ GPUTIMER::GPUTIMER(cudaStream_t stream_) : stream(stream_)
{
    gpuErrchk(cudaEventCreate(&start_event));
    gpuErrchk(cudaEventCreate(&stop_event));
}

__host__ GPUTIMER::~GPUTIMER()
{
    gpuErrchk(cudaEventDestroy(start_event));
    gpuErrchk(cudaEventDestroy(stop_event));
}

__host__ void GPUTIMER::start()
{
    gpuErrchk(cudaEventRecord(start_event, stream)); // Records an event.
}

__host__ void GPUTIMER::stop()
{
    gpuErrchk(cudaEventRecord(stop_event, stream)); // Records an event.
    gpuErrchk(cudaEventSynchronize(stop_event));    // Waits for an event to complete.
}

__host__ float GPUTIMER::get_time()
{
    // Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
    float elapsedTime;
    gpuErrchk(cudaEventElapsedTime(&elapsedTime, start_event, stop_event));
    return elapsedTime; // ms
}

__host__ std::string GPUTIMER::get_time_str(float elapsedTime)
{
    std::stringstream strstream;
    if (elapsedTime < 1000) // for example, 963.5ms
        strstream << std::setprecision(4) << elapsedTime << "ms";
    else if (elapsedTime < 60000) // for example, 9.635s
        strstream << std::setprecision(4) << static_cast<int>(elapsedTime) / 1000.0 << "s";
    else // for example, 3min48s
    {
        int min = static_cast<int>(elapsedTime / 60000);
        int sec = static_cast<int>(elapsedTime / 1000) - 60 * min;
        strstream << min << "min" << sec << "s";
    }
    return strstream.str();
}

__host__ std::string GPUTIMER::get_time_str()
{
    float elapsedTime = get_time();
    return get_time_str(elapsedTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ void checkDeviceInfo()
{
    // check existence
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount != 0)
        std::cout << "Find " << deviceCount << " GPU" << (deviceCount > 1 ? "s" : "") << " in your system" << std::endl;
    else
    {
        TORCH_CHECK(deviceCount != 0, "Find no GPU device in your system");
    }

    // print info
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "GPU:" << device
                  << " | " << deviceProp.name
                  << " | ComputeCapability: " << deviceProp.major << "." << deviceProp.minor
                  << " | GlobalMemorySize: " << deviceProp.totalGlobalMem << "bytes"
                  << std::endl;
    }

    // set device
    gpuErrchk(cudaSetDevice(0)); // setting GPU:0 as default, usually in `default compute mode`
    std::cout << "The GPU:0 is used for computation by default" << std::endl;

    // print memory info
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "Free: " << free_bytes << "bytes | Total: " << total_bytes << "bytes" << std::endl;

    // set heap size
    gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 * 1024 * 1024 * 1024));
    std::cout << "Set cudaLimitMallocHeapSize to 1GB" << std::endl;
}
