# How to build and use Tensorflow XLA/AOT on Windows

To use Tensorflow XLA/AOT on Windows, we need `tfcompile` XLA AOT compiler to
compile model into native code as well as some runtime libraries to build the
final executable.

Upstream tracking bug for Windows support is at https://github.com/tensorflow/tensorflow/issues/15213

Note that XLA/AOT itself is experimental, there are things that do not work regardless
of the target operating system. If you are sure the bug is only for Windows, comment at the
tracking bug above.

## Get Tensorflow source code

Currently XLA/AOT for Windows only works on the master branch

```
C:\>git clone --depth 1 https://github.com/tensorflow/tensorflow
C:\>cd tensorflow
```

## Configure Tensorflow

If you have not installed Bazel, download `bazel-0.17.1-windows-x86_64.exe`
from https://github.com/bazelbuild/bazel/releases, rename it to `bazel.exe` and put it in `PATH`.

Run `configure.py` (see https://www.tensorflow.org/install/install_sources).

Open `.tf_configure.bazelrc` with your text editor.

Add the following code to `.tf_configure.bazelrc` to set up global compile flags:

```
build --copt=-DNOGDI
build --host_copt=-DNOGDI
```

Now we can finally starts to build `tfcompile` with Bazel!

```
C:\tensorflow>bazel build -c opt --config=opt //tensorflow/compiler/aot:tfcompile
```

If everything goes well, you should have `tfcompile` installed at
`C:\tensorflow\bazel-bin\tensorflow\compiler\aot\tfcompile.exe`.

## Use Tensorflow XLA/AOT on Windows

You should read https://www.tensorflow.org/performance/xla/tfcompile to learn
how XLA/AOT works before continue with this tutorial.

Due to https://github.com/bazelbuild/bazel/issues/4149, we can't directly use
`tf_library` macro detailed in https://www.tensorflow.org/performance/xla/tfcompile, which means we need to
do a lot of things manually.

For this tutorial, I prepared a simple MNIST demo [`mnist.py`](mnist.py). You
should quickly scan through the code.

Run `mnist.py`. A few files will be generated:

- `mnist.pb`: Protobuf file that describe the graph
- `mnist.ckpt`: Checkpoint file that records the values of `tf.Variable` nodes.
- `mnist.ckpt.meta` and `checkpoint`: Metadata for checkpoint file.

Because our MNIST graph has some `tf.Variable` nodes, we need to "freeze" them
to be all constants with `freeze_grah.py`. The output node is named "y", so that
is the value of `--output_node_names`. 

```
C:\tensorflow\tensorflow\python\tools\freeze_graph.py --input_graph=mnist.pb ^
--input_binary=true --input_checkpoint=mnist.ckpt ^
--output_graph=mnist_freeze.pb --output_node_names=y
```

You should see `Converted 2 variables to const ops.`, which means two `tf.Variable` nodes named "w" and "b" are now freezed into constant nodes.

Now feed the freezed graph to `tfcompile`.

```
C:\tensorflow\bazel-bin\tensorflow\compiler\aot\tfcompile.exe ^
--graph=mnist_freeze.pb --config=mnist_config.pbtxt --entry_point=mnist_entry ^
--cpp_class=MnistGraph --target_triple=x86_64-windows-msvc ^
--out_header=mnist.h --out_object=mnist.lib
```

Note:

- `--target=x86_64-windows-msvc`: This tells LLVM backend (internal of
`tfcompile`) to generate 64-bit native code for Windows in MSVC ABI.
- `--config`, `--graph` and `--cpp_class` conrespond to the attributes of `tf_library` macro.
- Make sure you feed the freezed graph to `--graph`, not the original graph,
otherwise `tfcompile` will hard abort (which is equivalent to a crash on
Windows).

Now we have `mnist.lib` with the generated code and `mnist.h` to invoke the
generated code. I prepared a simple C++ code ['mnist.cpp`](mnist.cpp) that draws
a "1" on the canvas and let the model predicts what number it is.

Copy `mnist.h`, `mnist.lib` and `mnist.cpp` to `C:\tensorflow`. Switch current
directory to `C:\tensorflow`.

Copy the following code into `BUILD`.

```
cc_library(
  name = "mnist_xla_lib",
  hdrs = ["mnist.h"],
  srcs = ["mnist.lib"],
)

cc_binary(
  name = "mnist",
  srcs = ["mnist.cpp"],
  deps = [
    ":mnist_xla_lib",
    "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
    "//tensorflow/core:framework_lite",
    "//tensorflow/compiler/tf2xla/kernels:index_ops_kernel_argmax_float_1d",
    "//tensorflow/compiler/tf2xla/kernels:index_ops_kernel_argmax_float_2d",
    "//tensorflow/compiler/xla/service/cpu:runtime_conv2d",
    "//tensorflow/compiler/xla/service/cpu:runtime_matmul",
    "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d",
    "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
    "//third_party/eigen3",
  ],
)
```

Now build and run the C++ demo.

```
C:\tensorflow>bazel build -c opt --config=opt //:mnist
C:\tensorflow>bazel-bin\mnist.exe
```

You should see something like this ("1" should have highest prediction score):

```
0: -7
1: 7
2: 1
3: 1
4: -4
5: 1
6: 1
7: -3
8: 3
9: -0
Success
```