# comfy-kitchen's setup.py builds with `cmake --build --parallel <cpu_count>` (32 here) and ignores
# MAX_JOBS, which would run 32 nvcc over CUTLASS at once and reach for more RAM than this box has.
# Job pools are the only way to throttle it without patching; they need the Ninja generator.
set(CMAKE_JOB_POOLS compile=4 link=1)
set(CMAKE_JOB_POOL_COMPILE compile)
set(CMAKE_JOB_POOL_LINK link)
