#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cJSON.h"
#include <string.h>
#include <math.h>
#include <sys/stat.h> // For mkdir on Unix-like systems
#include <errno.h>    // For errno and EEXIST on Unix-like systems
#include <omp.h>      // Include OpenMP header

#ifdef _WIN32
#include <direct.h> // For mkdir on Windows
#define mkdir(dir) _mkdir(dir) // Define mkdir for Windows
#endif

#define PROGRESS_BAR_WIDTH 50  // Width of the progress bar

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

// Function to apply Gaussian filter with OpenMP parallelization
void applyGaussianFilter(unsigned char *image_data, int width, int height) {
    int kernel_size = 5;
    double sigma = 1.5;
    double kernel[5][5];
    double sum = 0.0;

    // Compute the Gaussian kernel (this is small and fast, so keep sequential)
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            double x = i - kernel_size / 2;
            double y = j - kernel_size / 2;
            kernel[i][j] = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++)
            kernel[i][j] /= sum;

    unsigned char *temp = (unsigned char *)malloc(width * height);
    if (!temp) return;

    int offset = kernel_size / 2;
    
    // Parallelize the main filtering loop
    #pragma omp parallel for schedule(dynamic)
    for (int i = offset; i < height - offset; i++) {
        for (int j = offset; j < width - offset; j++) {
            double pixel_value = 0.0;
            for (int k = -offset; k <= offset; k++) {
                for (int l = -offset; l <= offset; l++) {
                    pixel_value += image_data[(i + k) * width + (j + l)] * kernel[k + offset][l + offset];
                }
            }
            temp[i * width + j] = (unsigned char)(pixel_value < 0 ? 0 : (pixel_value > 255 ? 255 : pixel_value));
        }
    }

    // Copy border pixels (not processed by the filter)
    for (int i = 0; i < offset; i++) {
        for (int j = 0; j < width; j++) {
            temp[i * width + j] = image_data[i * width + j];
            temp[(height - i - 1) * width + j] = image_data[(height - i - 1) * width + j];
        }
    }
    
    for (int i = offset; i < height - offset; i++) {
        for (int j = 0; j < offset; j++) {
            temp[i * width + j] = image_data[i * width + j];
            temp[i * width + (width - j - 1)] = image_data[i * width + (width - j - 1)];
        }
    }

    memcpy(image_data, temp, width * height);
    free(temp);
}

// Function to apply Wiener filter (approximation) with OpenMP parallelization
void applyWienerFilter(unsigned char *image_data, int width, int height) {
    int kernel_size = 5;
    int kernel_area = kernel_size * kernel_size;

    unsigned char *temp = (unsigned char *)malloc(width * height);
    if (!temp) return;

    int offset = kernel_size / 2;
    
    // Parallelize the main filtering loop
    #pragma omp parallel for schedule(dynamic)
    for (int i = offset; i < height - offset; i++) {
        for (int j = offset; j < width - offset; j++) {
            int sum = 0;
            for (int k = -offset; k <= offset; k++) {
                for (int l = -offset; l <= offset; l++) {
                    sum += image_data[(i + k) * width + (j + l)];
                }
            }
            temp[i * width + j] = sum / kernel_area;
        }
    }

    // Copy border pixels (not processed by the filter)
    for (int i = 0; i < offset; i++) {
        for (int j = 0; j < width; j++) {
            temp[i * width + j] = image_data[i * width + j];
            temp[(height - i - 1) * width + j] = image_data[(height - i - 1) * width + j];
        }
    }
    
    for (int i = offset; i < height - offset; i++) {
        for (int j = 0; j < offset; j++) {
            temp[i * width + j] = image_data[i * width + j];
            temp[i * width + (width - j - 1)] = image_data[i * width + (width - j - 1)];
        }
    }

    memcpy(image_data, temp, width * height);
    free(temp);
}

// Function to process the dataset and save the processed images in a different folder
void processDataset(const char *json_path, const char *image_dir, const char *output_dir) {
    double start_time, end_time;

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

    // Create the output directory if it doesn't exist
    if (mkdir(output_dir) == -1 && errno != EEXIST) {
        printf("Error creating output directory: %s\n", output_dir);
        cJSON_Delete(root);
        free(json_data);
        return;
    }

    // Get number of available threads
    int num_threads = omp_get_max_threads();
    printf("Processing with %d OpenMP threads\n", num_threads);

    start_time = omp_get_wtime(); // Use OpenMP timing for more accuracy

    // Process images
    // Note: We don't parallelize this outer loop because file I/O might cause conflicts
    // Instead, we parallelize the filter application inside each image
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

        // Apply filters
        applyGaussianFilter(image_data, width, height);
        applyWienerFilter(image_data, width, height);

        // Save processed image to the new location
        stbi_write_png(output_path, width, height, 1, image_data, width);
        stbi_image_free(image_data);

        // Update progress bar (with thread safety)
        #pragma omp critical
        {
            processed_images++;
            printProgressBar(processed_images, total_images);
        }
    }

    cJSON_Delete(root);
    free(json_data);

    end_time = omp_get_wtime();
    
    printf("\nProcessing time: %.3f seconds\n", end_time - start_time);
}

int main() {
    printf("Starting parallelized model training...\n");

    // Set number of OpenMP threads (optional, can also be controlled by environment variable OMP_NUM_THREADS)
    // Uncomment and adjust if you want to specify a specific number of threads
    // omp_set_num_threads(4);

    processDataset("C:/Users/V Abhiram/Desktop/IT301M/project/SARscope/test/_annotations.coco.json", 
                   "C:/Users/V Abhiram/Desktop/IT301M/project/SARscope/test",
                   "C:/Users/V Abhiram/Desktop/IT301M/project/SARscope/test/processed_images");

    printf("\nTraining complete.\n");
    return 0;
}