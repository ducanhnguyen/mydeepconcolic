char *concat(const char *a, const char *b){
    int lena = strlen(a);
    int lenb = strlen(b);
    char *con = malloc(lena+lenb+1);
    // copy & concat (including string termination)
    memcpy(con,a,lena);
    memcpy(con+lena,b,lenb+1);
    return con;
}