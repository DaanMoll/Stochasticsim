ortho-sampling: ortho-sampling.c rand_support.c rand_support.h mt19937.c mt19937.h
	clang -fsanitize=signed-integer-overflow -fsanitize=undefined -ggdb3 -O0 -Qunused-arguments -std=c11 -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wshadow -o ortho ortho-sampling.c mt19937.c rand_support.c
