#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cJSON.h"

#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)
#endif

#define PROGRESS_BAR_WIDTH 50
#define BLOCK_SIZE 16  // CUDA block size (16x16 threads)

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Function to print the progress bar
void printProgressBar(int current, int total) {
    float percentage = (float)current / total;
    int completed = (int)(percentage * PROGRESS_BAR_WIDTH);
    
    printf("\r[");  
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++) {
        if (i < completed) printf("=");
        else printf(" ");
    }
    printf("] %d%%", (int)(percentage * 100));
    fflush(stdout);
}

// Function to process the dataset and save the processed images in a different folder
void processDataset(const char *json_path, const char *image_dir, const char *output_dir) {
    clock_t start, end;
    double cpu_time_used;

    FILE *file = fopen(json_path, "r");
    if (!file) {
        printf("Could not open JSON file: %s\n", json_path);
        return;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json_data = (char *)malloc(length + 1);
    fread(json_data, 1, length, file);
    json_data[length] = '\0';
    fclose(file);

    cJSON *root = cJSON_Parse(json_data);
    if (!root) {
        printf("Error parsing JSON\n");
        free(json_data);
        return;
    }

    cJSON *images = cJSON_GetObjectItem(root, "images");
    if (!cJSON_IsArray(images)) {
        printf("Invalid JSON format\n");
        cJSON_Delete(root);
        free(json_data);
        return;
    }

    int total_images = cJSON_GetArraySize(images);
    int processed_images = 0;

    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));

    start = clock(); // Start timing

    cJSON *image_item;
    cJSON_ArrayForEach(image_item, images) {
        cJSON *file_name = cJSON_GetObjectItem(image_item, "file_name");
        if (!cJSON_IsString(file_name)) continue;

        char image_path[512];
        char output_path[512];
        
        // Original image path
        sprintf(image_path, "%s/%s", image_dir, file_name->valuestring);
        
        // Output path for processed images
        sprintf(output_path, "%s/%s", output_dir, file_name->valuestring);

        int width, height, channels;
        unsigned char *image_data = stbi_load(image_path, &width, &height, &channels, STBI_grey);
        if (!image_data) {
            printf("\nCould not read image: %s\n", image_path);
            continue;
        }

        // Create the output directory if it doesn't exist
        #ifndef _WIN32
        mkdir(output_dir, 0777);
        #else
        mkdir(output_dir);
        #endif

        // Save processed image to the new location
        stbi_write_png(output_path, width, height, 1, image_data, width);
        stbi_image_free(image_data);

        // Update progress bar
        processed_images++;
        printProgressBar(processed_images, total_images);
    }

    cJSON_Delete(root);
    free(json_data);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("\nProcessing time: %.3f seconds\n", cpu_time_used);
}

int main() {
    printf("Starting CUDA-accelerated model training...\n");

    processDataset("/content/dataset/SARscope/test/_annotations.coco.json",
    "/content/dataset/SARscope/test",
    "/content/dataset/processed_images");

    printf("\nTraining complete.\n");
    
    // Cleanup CUDA resources
    cudaDeviceReset();
    
    return 0;
}
