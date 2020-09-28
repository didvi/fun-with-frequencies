# USAGE
Disclaimer: abandoned the main function since it was pretty hard to generalize. For each part, call the corresponding script. All scripts take in an image path (or sometimes two image paths), a boolean if you want to save the image, and a boolean if your image is in color or not.

## Part 1
```python gradients.py -i <img path> -f <function name>```
For function name, call dx, dy, gauss, dxog (derivative of gaussian), dyog, or grad_magnitude

## Part 2.1
```python sharpen.py -i <img path> -a <alpha level>```

## Part 2.2
```python combine.py -i <img path> -l <second image>```

## Part 2.3
```python freq_stack.py -i <img path>```

## Part 2.4
```python blend.py -i <img path> --img2 <second img>```
