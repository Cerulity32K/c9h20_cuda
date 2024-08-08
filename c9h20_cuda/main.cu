/// 
/// \/\/\/\/ Nonane (C9H20) \/\/\/\/
///  -~***~-  CUDA Edition  -~***~-
///	 -=* Created by Cerulity32K *=-
/// 
/// A small program meant to fiddle around and screw with the capabilities of WinGDI and CUDA, with some WaveOut included.
/// 
/// Heavily inspired by Trichloromethane.exe (aka Chloroform.exe), MEMZ, and similar malware.
/// 
/// While these programs are malware (as they overwrite the MBR), this is not.
/// This program is instead just meant to look cool and be what older programs used to be.
/// Although visually and audibly invasive, this program, ***as far as i know***, is completely safe to run.
/// 
/// Although this is a C++ program, it is styled and structured more in the direction of
/// how a C program would be structured, given the low-level nature of the program.
///
/// More accurately, it simply doesn't use heavy object-orientation, only functions and data.
/// The reason it is written in C++ is because there are existing, helpful standard library features used in non-kernels,
/// and NVCC natively compiles for C++, and it feels wasteful to not use what's available.
///
/// 
/// CURRENT ISSUES:
/// Some of the time, WaveOut may have a stroke and start reading the binary as PCM data.
/// This seems to be completely random, and I have no clue how to fix it.
/// This is not good, it's really noisy and causes a segfault upon reaching the end of the binary.
/// Anyone who might know how to fix it, please let me know. I might just switch to WASAPI or even SoLoud.
/// 
/// Changelog:
/// v1.0: Initial release, with 5 graphical and audio effects (<graphical effect> | <audio effect> | <duration>):
/// - Pixel dissolve | Sierpinski melody | 20s
/// - Rectangle jumble | 42 melody V2 | 16.5s
/// - Modified XOR fractal | Woimp | 10s
/// - Icon spam | Mosquito | 15s
/// - Text spam | Chip arp that eats itself | 12s
/// 
/// v2.0: Make program proper, faster, and flashier:
///	- Clean up threads the right way (events and signal checks instead of TerminateThread), allowing threads to clean up after themselves.
/// - Optimize the pixel dissolving effect and XOR fractal effect with CUDA.
/// - Replace rectangle jumbling with a quirky, CUDA-accelerated interlacing effect.
/// Replace icon spam with a blurring Von Neumann median effect.
/// - Reduce the amount of pixel drift in the pixel dissolving effect to give a blobbier effect, similar to GOL amoeba rules.
/// - Speed up text spam to keep up with the flashiness of the new CUDA-accelerated effects.
/// Change the order of the effects.

#pragma region Preprocessor
#include <iostream>
#include <algorithm>
#include <vector>

#include <windows.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>

#include <numeric>

#define AUDIO
#pragma endregion

#pragma region Globals
const int screen_width = GetSystemMetrics(0);
const int screen_height = GetSystemMetrics(1);
HANDLE graphics_terminate_event_handle;

// Some pre-chosen spam texts.
const char* TEXT_SPAM[] = {
	"cerulity32k.github.io",
	"who said malware cant be fun?",
	"recycle this trash world",
	"go make something, be creative <3",
	":3 :3 :3 :3 :3 :3 :3 :3",
	"meow meow meow meow meow meow meow",
	"visual c++? yeah this is pretty visible",
	"cuda? cudeez nuts lmao",
};
const size_t TEXT_SPAM_COUNT = sizeof(TEXT_SPAM) / sizeof(const char*);

// A pool of icons to randomly pick from.
static HICON ICON_POOL[] = {
	LoadIcon(nullptr, IDI_WARNING),
	LoadIcon(nullptr, IDI_ERROR),
	LoadIcon(nullptr, IDI_SHIELD),
	LoadIcon(nullptr, IDI_APPLICATION),
	LoadIcon(nullptr, IDI_HAND),
	LoadIcon(nullptr, IDI_INFORMATION),
	LoadIcon(nullptr, IDI_QUESTION),
};
const size_t ICON_POOL_COUNT = sizeof(ICON_POOL) / sizeof(HICON);
#pragma endregion

#pragma region Helpers

// throwing runtime assertion with a custom message
#define throw_assert(expr, msg) if (!(expr)) { throw std::runtime_error(std::string(msg)); }

// Can be used in place of RGBQUAD or COLORREF, allows for both packed and unpacked RGB manipulation.
typedef union rgb_t {
	COLORREF rgb;
	RGBQUAD quad;
	struct {
		BYTE r, g, b, reserved;
	};
} rgb;
static_assert(sizeof(rgb_t) == sizeof(RGBQUAD), "rgb_t has a different size than RGBQUAD");

// Useful for marking durations.
typedef DWORD MILLISECONDS;
// Buffer for `bytebeat`, must be freed after audio is done playing.
LPSTR bytebeat_buffer;
// Queues sound, where the sample source is a function that returns a sample given a sample index.
// That is the definition of Bytebeat, a type/genre of music.
HWAVEOUT bytebeat(DWORD samplerate, MILLISECONDS duration, CHAR(*source)(size_t)) {
	HWAVEOUT wave_out_handle = 0;
	WAVEFORMATEX wave_format = { WAVE_FORMAT_PCM, 1, samplerate, samplerate, 1, 8, 0 };
	waveOutOpen(&wave_out_handle, WAVE_MAPPER, &wave_format, 0, 0, CALLBACK_NULL);

	size_t buffer_size = (size_t)samplerate * (size_t)duration / 1000;
	bytebeat_buffer = new CHAR[buffer_size];
	for (size_t t = 0; t < buffer_size; ++t) {
		bytebeat_buffer[t] = source(t);
	}

	WAVEHDR header = { bytebeat_buffer, buffer_size * sizeof(CHAR), 0, 0, 0, 0, 0, 0 };
	waveOutPrepareHeader(wave_out_handle, &header, sizeof(WAVEHDR));
	waveOutWrite(wave_out_handle, &header, sizeof(WAVEHDR));
	waveOutUnprepareHeader(wave_out_handle, &header, sizeof(WAVEHDR));
	waveOutClose(wave_out_handle);
	return wave_out_handle;
}
// Releases the bytebeat buffer.
void bytebeat_destructor() {
	delete[] bytebeat_buffer;
}
// Stores a memory DC.
struct mem_dc {
	HDC dc;
	rgb_t* buffer_data;
	HBITMAP bitmap_handle;

	mem_dc(HDC memory_dc, RGBQUAD* buffer_data, HBITMAP bitmap_handle)
		: dc{ memory_dc }, buffer_data{ (rgb_t*)buffer_data }, bitmap_handle{ bitmap_handle } {}
};
// Creates a memory DC.
mem_dc make_memory_screen() {
	HDC main_screen = GetDC(0);
	HDC memory_dc = CreateCompatibleDC(main_screen);
	throw_assert(memory_dc, "could not create memory DC");
	BITMAPINFO bitmap_info = { 0 };
	bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFO);
	bitmap_info.bmiHeader.biBitCount = 32;
	bitmap_info.bmiHeader.biPlanes = 1;
	bitmap_info.bmiHeader.biWidth = screen_width;
	bitmap_info.bmiHeader.biHeight = screen_height;
	RGBQUAD* screen_data{ nullptr };
	HBITMAP bitmap_handle = CreateDIBSection(main_screen, &bitmap_info, 0, (void**)&screen_data, nullptr, 0);
	throw_assert(bitmap_handle, "could not create DIB section");
	SelectObject(memory_dc, bitmap_handle);
	ReleaseDC(nullptr, main_screen);

	return mem_dc{
		memory_dc,
		screen_data,
		bitmap_handle,
	};
}
// Destroys a memory DC.
void destroy_memory_screen(mem_dc& memory_dc) {
	DeleteDC(memory_dc.dc);
	DeleteObject(memory_dc.bitmap_handle);
}
// Parameters that are created and passed to every GDI manipulator thread.
// Mostly used to enforce lifetimes. In the future, destructor functions may be attached to GDI manipulators.
struct gdi_manipulator_data {};
bool graphics_should_run() {
	DWORD result = WaitForSingleObject(graphics_terminate_event_handle, 0);
	throw_assert(result != WAIT_FAILED, "failed to wait on graphics_terminate_event_handle");
	return result == WAIT_TIMEOUT; // timeout is good; handle was not already signalled or abandoned
}

// Only prints in debug mode.
template<typename T> void debug_log(T msg) {
#ifdef _DEBUG
	std::cout << msg;
#endif
}
#pragma endregion

#pragma region CUDA Kernels
// taken from stackoverflow, simple overflowing prng concept
__device__ size_t cuda_prng2(size_t x, size_t y) {
	x = x * 3266489917 + 374761393;
	x = (x << 17) | (x >> 15);

	x += y * 3266489917;

	x *= 668265263;
	x ^= x >> 15;
	x *= 2246822519;
	x ^= x >> 13;
	x *= 3266489917;
	x ^= x >> 16;

	return x;
}
__device__ size_t clamp_0_to(intptr_t x, size_t max) {
	if (x < 0) return 0;
	if (x > max) return max;
	return x;
}
__device__ size_t rem_euclid(intptr_t x, size_t max) {
	if (x < 0) return (x % max) + max;
	return x % max;
}
// Operates out-of-place; pixels are dependent on their neighbors
__global__ void cuda_pixel_dissolver(const rgb_t* src, rgb_t* dst, size_t frame, size_t dissolve_amount, int screen_width, int screen_height) {
	size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst_i >= screen_width * screen_height) return;

	size_t dst_x = dst_i % screen_width;
	size_t dst_y = dst_i / screen_width;

	intptr_t x_prng = cuda_prng2(dst_x, frame * 3659203 + dst_y) % (dissolve_amount * 2 + 1);
	intptr_t y_prng = cuda_prng2(dst_y, frame * 3659205 + dst_x) % (dissolve_amount * 2 + 1);

	intptr_t tmp_src_x = dst_x + x_prng - dissolve_amount;
	intptr_t tmp_src_y = dst_y + y_prng - dissolve_amount;

	size_t src_x = clamp_0_to(tmp_src_x, screen_width - 1);
	size_t src_y = clamp_0_to(tmp_src_y, screen_height - 1);

	size_t src_i = src_x + src_y * screen_width;

	dst[dst_i].rgb = src[src_i].rgb;
}
// Operates in-place; pixels only use themselves.
__global__ void cuda_xor_fractal(rgb_t* buf, int screen_width, int screen_height) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= screen_width * screen_height) return;

	size_t x = i % screen_width;
	size_t y = i / screen_width;

	buf[i].rgb += (x ^ y) ^ 1020;
	buf[i].rgb += 1234;
}
// Performs an odd interlacing effect, where every row moves opposite to its neighbors
// Operates out-of-place; pixels are dependent on pixels that come next
__global__ void cuda_interlace(const rgb_t* src, rgb_t* dst, int screen_width, int screen_height) {
	size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (dst_i >= screen_width * screen_height) return;

	size_t dst_x = dst_i % screen_width;
	size_t dst_y = dst_i / screen_width;

	intptr_t amount = (dst_y * dst_x * 25 / screen_height / screen_width) + 1;
	intptr_t tmp_src_x = (intptr_t)dst_x + (((intptr_t)dst_y & 1) ? -amount : amount);

	size_t src_x = rem_euclid(tmp_src_x, screen_width);
	size_t src_y = dst_y;

	size_t src_i = src_x + src_y * screen_width;

	dst[dst_i].rgb = src[src_i].rgb;
}
__device__ float luminance(rgb_t color) {
	return color.r * 0.00083372549 + color.g * 0.00280470588 + color.b * 0.00028313725;
}
// Performs a bubble sort on a set of colors.
// A more efficient sorting algorithm could be used,
// but the simplicity of bubble sort might actually make it faster
// on small datasets (~5 colors)
__device__ void sort_colors(rgb_t* data, size_t count) {
	for (size_t last_check = count - 1; last_check > 0; last_check--) {
		for (size_t i = 0; i < last_check; i++) {
			if (luminance(data[i]) > luminance(data[i + 1])) {
				rgb_t tmp = data[i];
				data[i] = data[i + 1];
				data[i + 1] = tmp;
			}
		}
	}
}
// Performs a Von Neumann median-like filter. Each pixel compares itself and its 4 adjacent neighbors to find a color to use.
// Operates out-of-place; pixels are dependent on their adjacent neighbors
__global__ void cuda_median(const rgb_t* src, rgb_t* dst, size_t frame_number, int screen_width, int screen_height) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= screen_width * screen_height) return;

	size_t x = i % screen_width;
	size_t y = i / screen_width;

	size_t left = clamp_0_to((intptr_t)x - 1, screen_width - 1) + y * screen_width;
	size_t right = clamp_0_to((intptr_t)x + 1, screen_width - 1) + y * screen_width;

	size_t top = x + clamp_0_to((intptr_t)y - 1, screen_height - 1) * screen_width;
	size_t bottom = x + clamp_0_to((intptr_t)y + 1, screen_height - 1) * screen_width;

	rgb_t color_set[5] = {
		src[i],
		src[left],
		src[right],
		src[top],
		src[bottom],
	};
	sort_colors(color_set, 5);
	dst[i] = color_set[(frame_number % 30) >= 15 ? 0 : 4];
}
#pragma endregion

#pragma region WinGDI Manipulators
// Performs a weird interlacing effect.
// Requires a CUDA-capable GPU.
DWORD gdi_interlace(gdi_manipulator_data* data) {
	debug_log("  Pixel dissolve\n");
	HDC main_screen = GetDC(0);
	mem_dc memory_screen = make_memory_screen();

	// allocate CUDA buffers
	rgb_t* src_buffer;
	rgb_t* dst_buffer;
	const int buffer_size = sizeof(rgb_t) * screen_width * screen_height;
	throw_assert(cudaMallocManaged(&src_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");
	throw_assert(cudaMallocManaged(&dst_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");


	// pull screen into ram
	BitBlt(memory_screen.dc, 0, 0, screen_width, screen_height, main_screen, 0, 0, SRCCOPY);

	// copy input screen data to managed memory block
	cudaMemcpy(src_buffer, memory_screen.buffer_data, buffer_size, cudaMemcpyDefault);

	while (graphics_should_run()) {

		// launch kernel
		int block_count = (screen_width * screen_height - 1) / 1024 + 1; // # of blocks required to operate on entire screen
		int threads_per_block = 1024; // max threads per block in CUDA
		cuda_interlace<<<block_count, threads_per_block>>>(src_buffer, dst_buffer, screen_width, screen_height);
		throw_assert(cudaDeviceSynchronize() == cudaSuccess, "failed to dissolve pixels with CUDA");

		// copy output screen data from managed memory block
		cudaMemcpy(memory_screen.buffer_data, dst_buffer, buffer_size, cudaMemcpyDefault);

		// push ram to screen
		BitBlt(main_screen, 0, 0, screen_width, screen_height, memory_screen.dc, 0, 0, SRCCOPY);

		// swap source and destination buffer pointers
		std::swap<rgb_t*>(src_buffer, dst_buffer);
	}

	debug_log("  Pixel dissolve finished\n");

	throw_assert(cudaFree(src_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");
	throw_assert(cudaFree(dst_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");

	destroy_memory_screen(memory_screen);
	ReleaseDC(nullptr, main_screen);
	return 0;

	// old code for gdi_rect_jumble
	/*debug_log("  Rectangle jumble\n");
	HDC main_screen = GetDC(0);
	while (graphics_should_run()) {
		int width = rand() % 100 + 50;
		int height = rand() % 100 + 50;

		int src_x = rand() % (screen_width - width);
		int src_y = rand() % (screen_height - height);
		int dst_x = rand() % (screen_width - width);
		int dst_y = rand() % (screen_height - height);

		BitBlt(main_screen, dst_x, dst_y, width, height, main_screen, src_x, src_y, SRCCOPY);
	}

	debug_log("  Rectangle jumble finished\n");
	ReleaseDC(nullptr, main_screen);*/
	return 0;
}
// Writes out a ton of spam text on the screen.
DWORD gdi_text_spam(gdi_manipulator_data* data) {
	debug_log("  Text spam\n");
	HDC dc = GetDC(0);

	const size_t text_per_sleep = 5;
	while (graphics_should_run()) {
		for (size_t i = 0; i < text_per_sleep; i++) {
			const char* text = TEXT_SPAM[rand() % TEXT_SPAM_COUNT];
			int dst_x = rand() % screen_width;
			int dst_y = rand() % screen_height;

			SetBkColor(dc, RGB(rand(), rand(), rand()));
			SetTextColor(dc, RGB(rand(), rand(), rand()));

			TextOutA(dc, dst_x, dst_y, text, strlen(text));
		}
	}

	debug_log("  Text spam finished\n");

	ReleaseDC(nullptr, dc);
	return 0;
}
// Creates a tweaked XOR fractal (Partially taken from Trichloromethane.exe).
// Requires a CUDA-capable GPU.
DWORD gdi_xoring(gdi_manipulator_data* data) {
	debug_log("  XOR fractal\n");
	HDC main_screen = GetDC(0);
	mem_dc memory_screen = make_memory_screen();

	// allocate CUDA buffer
	rgb_t* frame_buffer;
	const int buffer_size = sizeof(rgb_t) * screen_width * screen_height;
	throw_assert(cudaMallocManaged(&frame_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");

	// pull screen into ram
	BitBlt(memory_screen.dc, 0, 0, screen_width, screen_height, main_screen, 0, 0, SRCCOPY);

	// copy input screen data to managed memory block
	cudaMemcpy(frame_buffer, memory_screen.buffer_data, buffer_size, cudaMemcpyDefault);

	while (graphics_should_run()) {

		// launch kernel
		int block_count = (screen_width * screen_height - 1) / 1024 + 1; // # of blocks required to operate on entire screen
		int threads_per_block = 1024; // max threads per block in CUDA
		cuda_xor_fractal<<<block_count, threads_per_block>>>(frame_buffer, screen_width, screen_height);
		throw_assert(cudaDeviceSynchronize() == cudaSuccess, "failed to dissolve pixels with CUDA");

		// copy output screen data from managed memory block
		cudaMemcpy(memory_screen.buffer_data, frame_buffer, buffer_size, cudaMemcpyDefault);

		// push ram to screen
		BitBlt(main_screen, 0, 0, screen_width, screen_height, memory_screen.dc, 0, 0, SRCCOPY);
	}

	debug_log("  XOR fractal finished\n");

	throw_assert(cudaFree(frame_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");

	destroy_memory_screen(memory_screen);
	ReleaseDC(nullptr, main_screen);
	return 0;
}
// Dissolves/melts the screen by moving pixels.
// Requires a CUDA-capable GPU.
DWORD gdi_pixeldissolve(gdi_manipulator_data* data) {
	debug_log("  Pixel dissolve\n");
	HDC main_screen = GetDC(0);
	mem_dc memory_screen = make_memory_screen();

	// allocate CUDA buffers
	rgb_t* src_buffer;
	rgb_t* dst_buffer;
	const int buffer_size = sizeof(rgb_t) * screen_width * screen_height;
	throw_assert(cudaMallocManaged(&src_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");
	throw_assert(cudaMallocManaged(&dst_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");
	
	const intptr_t drift_amount = 5;
	size_t frame_number = 0;


	// pull screen into ram
	BitBlt(memory_screen.dc, 0, 0, screen_width, screen_height, main_screen, 0, 0, SRCCOPY);

	// copy input screen data to managed memory block
	cudaMemcpy(src_buffer, memory_screen.buffer_data, buffer_size, cudaMemcpyDefault);

	while (graphics_should_run()) {

		// launch kernel
		int block_count = (screen_width * screen_height - 1) / 1024 + 1; // # of blocks required to operate on entire screen
		int threads_per_block = 1024; // max threads per block in CUDA
		cuda_pixel_dissolver<<<block_count, threads_per_block>>>(src_buffer, dst_buffer, frame_number, drift_amount, screen_width, screen_height);
		throw_assert(cudaDeviceSynchronize() == cudaSuccess, "failed to perform CUDA XOR fractal");

		// copy output screen data from managed memory block
		cudaMemcpy(memory_screen.buffer_data, dst_buffer, buffer_size, cudaMemcpyDefault);

		// push ram to screen
		BitBlt(main_screen, 0, 0, screen_width, screen_height, memory_screen.dc, 0, 0, SRCCOPY);

		// swap source and destination buffer pointers
		std::swap<rgb_t*>(src_buffer, dst_buffer);
		frame_number++;
	}

	debug_log("  Pixel dissolve finished\n");

	throw_assert(cudaFree(src_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");
	throw_assert(cudaFree(dst_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");

	destroy_memory_screen(memory_screen);
	ReleaseDC(nullptr, main_screen);
	return 0;
}
// Performs a Von Neumann median effect using the 4 neighbors of each pixel.
// Requires a CUDA-capable GPU.
DWORD gdi_median(gdi_manipulator_data* data) {
	debug_log("  Median\n");
	HDC main_screen = GetDC(0);
	mem_dc memory_screen = make_memory_screen();

	// allocate CUDA buffers
	rgb_t* src_buffer;
	rgb_t* dst_buffer;
	const int buffer_size = sizeof(rgb_t) * screen_width * screen_height;
	throw_assert(cudaMallocManaged(&src_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");
	throw_assert(cudaMallocManaged(&dst_buffer, buffer_size) == cudaSuccess, "failed to allocate CUDA managed buffer memory");


	// pull screen into ram
	BitBlt(memory_screen.dc, 0, 0, screen_width, screen_height, main_screen, 0, 0, SRCCOPY);

	// copy input screen data to managed memory block
	cudaMemcpy(src_buffer, memory_screen.buffer_data, buffer_size, cudaMemcpyDefault);

	size_t frame_number = 0;

	while (graphics_should_run()) {

		// launch kernel
		int block_count = (screen_width * screen_height - 1) / 1024 + 1; // # of blocks required to operate on entire screen
		int threads_per_block = 1024; // max threads per block in CUDA
		cuda_median<<<block_count, threads_per_block>>>(src_buffer, dst_buffer, frame_number, screen_width, screen_height);
		throw_assert(cudaDeviceSynchronize() == cudaSuccess, "failed to perform CUDA median effect");

		// copy output screen data from managed memory block
		cudaMemcpy(memory_screen.buffer_data, dst_buffer, buffer_size, cudaMemcpyDefault);

		// push ram to screen
		BitBlt(main_screen, 0, 0, screen_width, screen_height, memory_screen.dc, 0, 0, SRCCOPY);

		// swap source and destination buffer pointers
		std::swap<rgb_t*>(src_buffer, dst_buffer);

		frame_number++;
	}

	debug_log("  Median finished\n");

	throw_assert(cudaFree(src_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");
	throw_assert(cudaFree(dst_buffer) == cudaSuccess, "failed to free CUDA managed buffer memory");

	destroy_memory_screen(memory_screen);
	ReleaseDC(nullptr, main_screen);
	return 0;
}
// Draws icons all over the screen.
DWORD gdi_icons(gdi_manipulator_data* data) {
	debug_log("Icons\n");

	HDC main_screen = GetDC(0);

	while (graphics_should_run()) {
		DrawIcon(main_screen, rand() % screen_width, rand() % screen_height, ICON_POOL[rand() % ICON_POOL_COUNT]);
	}

	debug_log("Icons finished\n");
	ReleaseDC(nullptr, main_screen);
	return 0;
}
#pragma endregion

#pragma region Audio Queuers
// Sierpinski melody (bytebeat)
// Composed by miiro (https://youtu.be/qlrs2Vorw2Y?t=2m14s)
// Found on dollchan.net (https://dollchan.net/bytebeat/)
HWAVEOUT waveout_sierpinski_melody(MILLISECONDS duration) {
	debug_log("  Queueing sierpinski melody...\n");
	return bytebeat(8000, duration, [](size_t t) -> CHAR { return 5 * t & t >> 7 | 3 * t & 4 * t >> 10; });
}
// The 42 melody v2 (bytebeat)
// Composed by viznut (http://viznut.fi/demos/unix/bytebeat_formulas.txt)
// Modified by me
// Found on dollchan.net (https://dollchan.net/bytebeat/)
HWAVEOUT waveout_42_melody_v2(MILLISECONDS duration) {
	debug_log("  Queueing 42 melody V2...\n");
	return bytebeat(8000, duration, [](size_t t) -> CHAR { return ((((t >> 14) % 8) + 1) * t * (((t >> 10) & (2 | 8) ^ 10))) >> 1; });
}
// Chip arpeggio that eats itself (bytebeat)
// Composed by kb_ (https://www.pouet.net/topic.php?which=8357&page=8#c388898)
// Found on dollchan.net (https://dollchan.net/bytebeat/)
HWAVEOUT waveout_chip_arp(MILLISECONDS duration) {
	debug_log("  Queueing chip arp that eats itself...\n");
	return bytebeat(44100, duration, [](size_t t) -> CHAR {
		return ((t >> 1) * (15 & 0x234568a0 >> (t >> 8 & 28)) | t >> 1 >> (t >> 11) ^ t >> 12) + (t >> 4 & t & 24);
	});
}
// Woimp (bytebeat)
// Composed by me (at least, according to the 3-month old list of bytebeats i have)
HWAVEOUT waveout_woimp(MILLISECONDS duration) {
	debug_log("  Queueing woimp...\n");
	return bytebeat(8000, duration, [](size_t t) -> CHAR {
		return t * ((((t >> 11) % 32) + 1) << ((t >> 8) % 8));
	});
}
// Mosquito (bytebeat)
// Composed by me
HWAVEOUT waveout_mosquito(MILLISECONDS duration) {
	debug_log("  Queueing mosquito...\n");
	return bytebeat(8000, duration, [](size_t t) -> CHAR {
		return t * ((((t >> 10) ^ 20) ^ (t >> 9)) & 0x1f);
	});
}
#pragma endregion

#pragma region Main
// An element in an effect sequence.
// Not a payload in the sense of an overwrite of some critical data, instead it is a visual/auditory payload.
struct payload {
	// The graphics function to be dispatched to a separate thread.
	// Should run until graphics_terminate_event_handle is signalled,
	// at which point the function should clean up.
	// 
	// Should also be formed like LPTHREAD_START_ROUTINE.
	DWORD(WINAPI* graphics)(gdi_manipulator_data*);
	// The audio queuer function.
	// This should return as soon as possible and queue up the given number of milliseconds of audio.
	HWAVEOUT(*audio_queuer)(MILLISECONDS);
	// Runs after audio is finished.
	void(*audio_destructor)();
	// The amount of time to run the payload for.
	MILLISECONDS duration;

	// Runs the graphics and audio associated with this effect payload.
	void run() {
		// queue up audio
		debug_log("Queueing audio...\n");
#ifdef AUDIO
		HWAVEOUT audio_handle = audio_queuer(duration);
#endif
		debug_log("Audio queued.\n");

		// set up thread and synchronization event
		debug_log("Launching graphics thread...\n");
		gdi_manipulator_data data{};
		graphics_terminate_event_handle = CreateEvent(nullptr, TRUE, FALSE, TEXT("stop_graphics_event"));
		throw_assert(graphics_terminate_event_handle, "graphics termination event could not be created");
		debug_log("Event created...\n");

		HANDLE graphics_thread = CreateThread(nullptr, NULL, (LPTHREAD_START_ROUTINE)graphics, &data, NULL, nullptr);
		throw_assert(graphics_thread, "graphics thread could not be created");
		debug_log("Graphics thread launched. Sleeping...\n");

		// wait until done
		Sleep(duration);

		// clean up
		debug_log("Cleaning up everything...\n");
#ifdef AUDIO
		waveOutClose(audio_handle);
		audio_destructor();
#endif
		throw_assert(SetEvent(graphics_terminate_event_handle), "failed to set graphics event");
		debug_log("Event signalled...\n");
		throw_assert(WaitForSingleObject(graphics_thread, INFINITE) == WAIT_OBJECT_0, "failed to wait for graphics thread");
		debug_log("Thread exited...\n");
		CloseHandle(graphics_thread);
		CloseHandle(graphics_terminate_event_handle);
		debug_log("Cleaned up, ready for next payload.\n");
	}
};

void check_gpu_capabilities() {
	cudaDeviceProp properties;
	if (cudaSetDevice(0) != cudaSuccess) {
		std::cerr << "No CUDA-capable GPU could be found!";
		Sleep(10000);
		return;
	}
	if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
		std::cerr << "Could not get GPU CUDA properties!";
		Sleep(10000);
		return;
	}
	if (!properties.unifiedAddressing) {
		std::cerr << "GPU does not have unified addressing!";
		Sleep(10000);
		return;
	}
}

int main() {
	srand(time(nullptr));
	check_gpu_capabilities();
	payload payloads[] = {
		payload{ gdi_pixeldissolve, waveout_sierpinski_melody, bytebeat_destructor, 16500 },
		payload{ gdi_interlace, waveout_42_melody_v2, bytebeat_destructor, 16500 },
		payload{ gdi_median, waveout_mosquito, bytebeat_destructor, 5000 },
		payload{ gdi_xoring, waveout_woimp, bytebeat_destructor, 10000, },
		payload{ gdi_text_spam, waveout_chip_arp, bytebeat_destructor, 12000 },
	};
	for (payload& payload : payloads) {
		payload.run();
	}
}
int WinMain(HINSTANCE, HINSTANCE, LPSTR, int) { main(); }
#pragma endregion
