# Organization

Options like molecule, grid size, etc, are compile time options.

Use https://github.com/yaml/libyaml to parse YAML in C for options related to molecule, etc.

Use argp to parse command line options https://stackoverflow.com/questions/9642732/parsing-command-line-arguments, https://www.gnu.org/software/libc/manual/html_node/Argp.html

The main purpose of this GPU ray-tracer is to read in a set of disk parameters and generate a set of channel maps.

In the spirit of maximizing the usage of the GPU, I think we should defer the pixel checking to each GPU. Basically, as a first iteration, just launch a bunch of kernels, one-per-pixel. Later, we can tweak this if necessary.

But, how should the blocks, and shared memory, be organized?

Design decision: Generate the 2D grid of DeltaV2, Upsilon, and S on the host, and then store it as a texture on the device.

After the results of a pixel trace operation, store the result of tau and I in shared memory, and then later bind this up into a larger image.

What is done on the kernel?

1) check to see if this pixel should be traced
2) determine the bounding zps
3) trace pixel, and deliver tau, I

Can use the trick for the 3/4 and 4/3 roots in the best practices guide, pg. 50/51.

Design decision: How do we block up the shared memory? Per channel? Per quadrant of the disk?
Reassemble image and transfer to host.
