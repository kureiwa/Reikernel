#include <crash.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void)
{
    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);
    char tmpfile[] = "/tmp/libcrash_check_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0) { perror("mkstemp"); return 1; }
    unlink(tmpfile);
    pid_t pid = fork();
    if (pid == 0) {
        static char buf[8192 + CRASH_MAX_USER_BLOBS *
                              (CRASH_MAX_BLOB_KEY + 8 + CRASH_MAX_BLOB_SIZE)];
        if (crash_install(buf, sizeof(buf), fd, NULL, 0,
                          CRASH_AFTER_RERAISE) != CRASH_OK) {
            _exit(100);
        }
        volatile int *p = NULL;
        *p = 42;
        _exit(99);
    }
    int status;
    waitpid(pid, &status, 0);
    struct stat st;
    fstat(fd, &st);
    printf("dump_size=%ld sizeof(crash_dump_t)=%zu\n",
           (long)st.st_size, sizeof(crash_dump_t));
    close(fd);
    return 0;
}
