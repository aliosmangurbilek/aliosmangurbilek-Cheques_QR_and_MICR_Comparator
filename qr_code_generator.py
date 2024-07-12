import qrcode

img = qrcode.make('bu bir test qr kodudur.')
type(img)
img.save("some_file2.png")