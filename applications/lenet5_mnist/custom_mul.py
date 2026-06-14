# custom_mul.py

CUSTOM_MUL_CODE = r'''
// Core Mantissa Approximation Logic (Levels 0-5)
__device__ inline float approx_t_mantissa(float x, float y, int level) {
    // Level 0: The base linear approximation
    float val = 1.5f * x + 1.5f * y - 2.25f;
    
    if (level >= 1) {
        float term = (x - 1.5f) / 4.0f;
        val += (y > 1.5f) ? term : -term;
    }
    if (level >= 2) {
        float b, s;
        if (x < 1.5f && y < 1.5f) { b = 1.25f; s = -1.0f; }
        else if (x < 1.5f && y > 1.5f) { b = 1.75f; s = -1.0f; }
        else if (x > 1.5f && y < 1.5f) { b = 1.25f; s = 1.0f; }
        else { b = 1.75f; s = 1.0f; }
        val += s * (y - b) / 8.0f;
    }
    if (level >= 3) {
        float step = 1.0f / 8.0f;
        int idx = (int)((x - 1.0f) / step);
        if (idx < 0) idx = 0; if (idx > 7) idx = 7;
        float a = 1.0f + step * (idx + 0.5f);
        val += ((y > 1.5f) ? 1.0f : -1.0f) * (x - a) / 16.0f;
    }
    if (level >= 4) {
        float step = 1.0f / 16.0f;
        int idx = (int)((y - 1.0f) / step);
        if (idx < 0) idx = 0; if (idx > 15) idx = 15;
        float b = 1.0f + step * (idx + 0.5f);
        val += ((x > 1.5f) ? 1.0f : -1.0f) * (y - b) / 32.0f;
    }
    if (level >= 5) {
        float step = 1.0f / 32.0f;
        int idx = (int)((x - 1.0f) / step);
        if (idx < 0) idx = 0; if (idx > 31) idx = 31;
        float a = 1.0f + step * (idx + 0.5f);
        val += ((y > 1.5f) ? 1.0f : -1.0f) * (x - a) / 64.0f;
    }
    return val;
}

// Reconstructs the approximate float value
__device__ inline float custom_mul_approx_fp32(float a, float b, int level) {
    if (a == 0.0f || b == 0.0f) return 0.0f;
    
    float sign = ((a < 0) ^ (b < 0)) ? -1.0f : 1.0f;
    int ex, ey;
    
    // frexp gives mantissa in [0.5, 1). Multiply by 2 to get [1, 2)
    float mx = frexpf(fabsf(a), &ex) * 2.0f;
    float my = frexpf(fabsf(b), &ey) * 2.0f;
    
    float mant_prod = approx_t_mantissa(mx, my, level);
    
    // Reconstruct: sign * mant_prod * 2^(ex + ey - 2)
    // We subtract 2 because we scaled both mx and my by 2 (ex+ey-2)
    return sign * mant_prod * powf(2.0f, (float)(ex + ey - 2));
}
'''
