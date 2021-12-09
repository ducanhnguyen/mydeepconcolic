char cwd[1000]; getcwd(cwd, sizeof(cwd));
//printf(cwd);

AKA_TEST_PATH_FOLDER = concat(cwd, "/");
AKA_TEST_PATH_FOLDER = concat(AKA_TEST_PATH_FOLDER, argv[argc - 1]);
AKA_TEST_PATH_FOLDER = concat(AKA_TEST_PATH_FOLDER, "/");

AKA_TEST_PATH_FILE = concat(AKA_TEST_PATH_FOLDER, argv[argc - 1]);
AKA_TEST_PATH_FILE = concat(AKA_TEST_PATH_FILE, ".txt");
AKA_TEST_PATH_FILE[strlen(AKA_TEST_PATH_FILE)] = '\0';
//printf("\n");
//printf(AKA_TEST_PATH_FILE);
//printf("\n");

struct stat st = {0};
if (stat(AKA_TEST_PATH_FOLDER, &st) == -1) {
    mkdir(AKA_TEST_PATH_FOLDER, 0700);
}