import qrcode
img = qrcode.make('bu bir test qr kodudur.')
type(img)  # qrcode.image.pil.PilImage
img.save("some_file2.png")