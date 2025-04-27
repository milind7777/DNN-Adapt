#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

struct MappedBin {
    float* data_ptr = nullptr;
    void* raw_ptr = nullptr;
    size_t file_size = 0;
    size_t num_elements = 0;
};

MappedBin mmap_bin_file(const std::string& bin_file_path) {
    auto fd = open(bin_file_path.c_str(), O_RDONLY);
    if(fd < 0) throw std::runtime_error("Failed to open bin file for mmap: " + bin_file_path);

    struct stat sb;
    if(fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to stat bin file: " + bin_file_path);
    }

    size_t file_size = sb.st_size;
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if(mapped == MAP_FAILED) throw std::runtime_error("Failed to mmap bin file: " + bin_file_path);

    return MappedBin {
        .data_ptr = reinterpret_cast<float*>(mapped),
        .raw_ptr = mapped,
        .file_size = file_size,
        .num_elements = file_size / sizeof(float)
    };
}



