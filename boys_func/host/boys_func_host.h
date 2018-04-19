#ifndef __BOYS_FUNC_HOST_H__
#define __BOYS_FUNC_HOST_H__

#include "../device/vector_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void boys_function_host(int order, FLOAT_TYPE * restrict x, FLOAT_TYPE * restrict F);

#ifdef __cplusplus
}
#endif

#endif
