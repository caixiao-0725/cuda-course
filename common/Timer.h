#include <windows.h>

class Timer
{
public:
    Timer() {
        QueryPerformanceFrequency(&tc);
    }

    void Start() {
        QueryPerformanceFrequency(&tc);
        QueryPerformanceCounter(&t1);
    }
    void Stop() {
        QueryPerformanceCounter(&t2);
    }

    double GetDeltaTime() {
        time = (double)(t2.QuadPart - t1.QuadPart) / (double)tc.QuadPart;
        return time;
    }

    LARGE_INTEGER t1, t2, tc;
    double time;
};