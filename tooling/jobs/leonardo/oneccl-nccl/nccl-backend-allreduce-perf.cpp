#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

namespace {

size_t parse_size(const std::string& input) {
    if (input.empty()) {
        throw std::invalid_argument("empty size");
    }

    size_t pos = 0;
    double value = std::stod(input, &pos);
    size_t multiplier = 1;

    if (pos < input.size()) {
        std::string suffix = input.substr(pos);
        std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (suffix == "k" || suffix == "kb") {
            multiplier = 1024;
        }
        else if (suffix == "m" || suffix == "mb") {
            multiplier = 1024 * 1024;
        }
        else if (suffix == "g" || suffix == "gb") {
            multiplier = 1024 * 1024 * 1024;
        }
        else {
            throw std::invalid_argument("unsupported size suffix: " + suffix);
        }
    }

    return static_cast<size_t>(value * static_cast<double>(multiplier));
}

std::vector<size_t> parse_counts(const std::string& input) {
    std::vector<size_t> values;
    std::stringstream ss(input);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(parse_size(item));
        }
    }

    if (values.empty()) {
        throw std::invalid_argument("no element counts specified");
    }

    return values;
}

sycl::device get_device_for_rank(int rank) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        throw std::runtime_error("no GPU devices found");
    }
    return devices[static_cast<size_t>(rank) % devices.size()];
}

void usage(const char* name) {
    std::cerr << "Usage: " << name
              << " [--elem_counts counts] [--iters n] [--warmup_iters n] [--check last|off]\n"
              << "  --elem_counts accepts comma-separated float element counts, e.g. 1024,4096,1M\n";
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string elem_counts_arg = "1024,4096,16384,65536,262144,1048576,4194304,16777216";
    int iters = 50;
    int warmup_iters = 10;
    bool check = true;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--elem_counts" && i + 1 < argc) {
                elem_counts_arg = argv[++i];
            }
            else if (arg == "--iters" && i + 1 < argc) {
                iters = std::stoi(argv[++i]);
            }
            else if (arg == "--warmup_iters" && i + 1 < argc) {
                warmup_iters = std::stoi(argv[++i]);
            }
            else if (arg == "--check" && i + 1 < argc) {
                std::string mode = argv[++i];
                check = (mode != "off");
            }
            else if (arg == "--help" || arg == "-h") {
                if (rank == 0) {
                    usage(argv[0]);
                }
                MPI_Finalize();
                return 0;
            }
            else {
                if (rank == 0) {
                    std::cerr << "unknown or incomplete argument: " << arg << "\n";
                    usage(argv[0]);
                }
                MPI_Finalize();
                return 2;
            }
        }

        if (iters < 1 || warmup_iters < 0) {
            throw std::invalid_argument("iters must be >= 1 and warmup_iters must be >= 0");
        }

        auto elem_counts = parse_counts(elem_counts_arg);

        ccl::init();

        sycl::device dev = get_device_for_rank(rank);
        sycl::context ctx(dev);
        sycl::queue queue(ctx, dev, sycl::property::queue::in_order());

        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type addr;
        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            addr = kvs->get_address();
            MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(addr);
        }

        auto ccl_dev = ccl::create_device(queue.get_device());
        auto ccl_ctx = ccl::create_context(queue.get_context());
        auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
        auto stream = ccl::create_stream(queue);

        if (rank == 0) {
            std::cout << "# oneCCL NCCL backend allreduce perf\n";
            std::cout << "# ranks " << size << " iters " << iters
                      << " warmup_iters " << warmup_iters << " check " << (check ? "last" : "off")
                      << "\n";
            std::cout << "# count bytes avg_us algbw_GBps busbw_GBps check\n";
        }

        for (size_t count : elem_counts) {
            std::vector<float> host_send(count, static_cast<float>(rank + 1));
            std::vector<float> host_recv(check ? count : 1, 0.0f);

            float* d_send = sycl::malloc_device<float>(count, queue);
            float* d_recv = sycl::malloc_device<float>(count, queue);
            if (!d_send || !d_recv) {
                throw std::runtime_error("sycl::malloc_device failed");
            }

            queue.memcpy(d_send, host_send.data(), count * sizeof(float)).wait();
            queue.memset(d_recv, 0, count * sizeof(float)).wait();

            for (int i = 0; i < warmup_iters; ++i) {
                ccl::allreduce(d_send,
                               d_recv,
                               count,
                               ccl::datatype::float32,
                               ccl::reduction::sum,
                               comm,
                               stream)
                    .wait();
            }
            queue.wait();
            MPI_Barrier(MPI_COMM_WORLD);

            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                ccl::allreduce(d_send,
                               d_recv,
                               count,
                               ccl::datatype::float32,
                               ccl::reduction::sum,
                               comm,
                               stream)
                    .wait();
            }
            queue.wait();
            MPI_Barrier(MPI_COMM_WORLD);
            auto stop = std::chrono::steady_clock::now();

            double local_us = std::chrono::duration<double, std::micro>(stop - start).count() /
                              static_cast<double>(iters);
            double avg_us = 0.0;
            MPI_Reduce(&local_us, &avg_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            int local_ok = 1;
            if (check) {
                queue.memcpy(host_recv.data(), d_recv, count * sizeof(float)).wait();
                float expected = static_cast<float>((size * (size + 1)) / 2);
                for (size_t i = 0; i < count; ++i) {
                    if (std::fabs(host_recv[i] - expected) > 1e-5f) {
                        local_ok = 0;
                        break;
                    }
                }
            }

            int global_ok = 0;
            MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

            if (rank == 0) {
                double bytes = static_cast<double>(count * sizeof(float));
                double algbw = bytes / avg_us / 1000.0;
                double busbw = algbw * (2.0 * static_cast<double>(size - 1) / static_cast<double>(size));
                std::cout << count << " " << static_cast<size_t>(bytes) << " " << avg_us << " "
                          << algbw << " " << busbw << " "
                          << (global_ok ? "ok" : "FAILED") << "\n";
            }

            sycl::free(d_send, queue);
            sycl::free(d_recv, queue);

            if (!global_ok) {
                MPI_Finalize();
                return 1;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
