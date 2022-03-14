Mujoco Envs


Errors while Installation

For Error:
   Exception: 
   Missing path to your environment variable. 
   Current values LD_LIBRARY_PATH=
   Please add following line to .bashrc:
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgwak1/.mujoco/mujoco210/bin
Do:
   source ~/.bashrc 

For Error:
  creating /tmp/pip-req-build-ybqkruux/mujoco_py/generated/_pyxbld_2.0.2.13_38_linuxgpuextensionbuilder/lib.linux-x86_64-3.8/mujoco_py
  x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 /tmp/pip-req-build-ybqkruux/mujoco_py/generated/_pyxbld_2.0.2.13_38_linuxgpuextensionbuilder/temp.linux-x86_64-3.8/tmp/pip-req-build-ybqkruux/mujoco_py/cymj.o /tmp/pip-req-build-ybqkruux/mujoco_py/generated/_pyxbld_2.0.2.13_38_linuxgpuextensionbuilder/temp.linux-x86_64-3.8/tmp/pip-req-build-ybqkruux/mujoco_py/gl/eglshim.o -L/home/disc/h.bonnavaud/.mujoco/mujoco210/bin -Wl,--enable-new-dtags,-R/home/disc/h.bonnavaud/.mujoco/mujoco210/bin -lmujoco210 -lglewegl -o /tmp/pip-req-build-ybqkruux/mujoco_py/generated/_pyxbld_2.0.2.13_38_linuxgpuextensionbuilder/lib.linux-x86_64-3.8/mujoco_py/cymj.cpython-38-x86_64-linux-gnu.so -fopenmp
  error: [Errno 2] No such file or directory: 'patchelf'
Do:
   sudo apt-get install patchelf