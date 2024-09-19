#ifndef KERNELS_COMMONS_H_
#define KERNELS_COMMONS_H_


#define do_not_optimize(val) asm volatile("" : : "x"(val) : "memory")
#define unused(val) (void)val

#endif //KERNERLS_COMMONS_H_H
