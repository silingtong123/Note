- example.c
  ```C++
  #include <time.h>
  double My_variable = 3.0;
  int fact(int n){
    if (n <= 1) return 1;
    else return n*fact(n-1);
  }
  int my_mod(int x, int y){
    return x%y;
  }
  char* get_time(){
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);
  }
  ```
- example.i
  ```swig
  %module example
   %{
   extern double My_variable;
   extern int fact(int n);
   extern int my_mod(int x, int y);
   extern char *get_time();
   %}
  extern double My_variable;
  extern int fact(int n);
  extern int my_mod(int x, int y);
  extern char* get_time();
  ```
- run.sh
  ```bash
  apt-get install swig
  swig -python example.i
  gcc -fpic -c example.c example_wrap.c -I/usr/include/python2.7
  ld -shared example.o example_wrap.o -o _example.so
  # python
  import example
  ```