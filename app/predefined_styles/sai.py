from app.predefined_styles.schemas import StyleSchema

sai_prompt = [
    StyleSchema(
        name='SAI 3D Model',
        positive='professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting',
        negative='ugly, deformed, noisy, low poly, blurry, painting',
        image='styles/sai/3d_model.jpg',
    ),
    StyleSchema(
        name='SAI Analog Film',
        positive='analog film photo {prompt} . faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage',
        negative='painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured',
        image='styles/sai/analog_film.jpg',
    ),
    StyleSchema(
        name='SAI Anime',
        positive='anime artwork {prompt} . anime style, key visual, vibrant, studio anime, highly detailed',
        negative='photo, deformed, black and white, realism, disfigured, low contrast',
        image='styles/sai/anime.jpg',
    ),
    StyleSchema(
        name='SAI Cinematic',
        positive='cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy',
        negative='anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured',
        image='styles/sai/cinematic.jpg',
    ),
    StyleSchema(
        name='SAI Comic Book',
        positive='comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
        negative='photograph, deformed, glitch, noisy, realistic, stock photo',
        image='styles/sai/comic_book.jpg',
    ),
    StyleSchema(
        name='SAI Craft Clay',
        positive='play-doh style {prompt} . sculpture, clay art, centered composition, Claymation',
        negative='sloppy, messy, grainy, highly detailed, ultra textured, photo',
        image='styles/sai/craft_clay.jpg',
    ),
    StyleSchema(
        name='SAI Digital Art',
        positive='concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed',
        negative='photo, photorealistic, realism, ugly',
        image='styles/sai/digital_art.jpg',
    ),
    StyleSchema(
        name='SAI Enhance',
        positive='breathtaking {prompt} . award-winning, professional, highly detailed',
        negative='ugly, deformed, noisy, blurry, distorted, grainy',
        image='styles/sai/enhance.jpg',
    ),
    StyleSchema(
        name='SAI Fantasy Art',
        positive='ethereal fantasy concept art of {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy',
        negative='photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white',
        image='styles/sai/fantasy_art.jpg',
    ),
    StyleSchema(
        name='SAI Isometric',
        positive='isometric style {prompt} . vibrant, beautiful, crisp, detailed, ultra detailed, intricate',
        negative='deformed, mutated, ugly, disfigured, blur, blurry, noise, noisy, realistic, photographic',
        image='styles/sai/isometric.jpg',
    ),
    StyleSchema(
        name='SAI Line Art',
        positive='line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics',
        negative='anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic',
        image='styles/sai/line_art.jpg',
    ),
    StyleSchema(
        name='SAI Lowpoly',
        positive='low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition',
        negative='noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo',
        image='styles/sai/lowpoly.jpg',
    ),
    StyleSchema(
        name='SAI Neonpunk',
        positive='neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional',
        negative='painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured',
        image='styles/sai/neonpunk.jpg',
    ),
    StyleSchema(
        name='SAI Origami',
        positive='origami style {prompt} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition',
        negative='noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo',
        image='styles/sai/origami.jpg',
    ),
    StyleSchema(
        name='SAI Photographic',
        positive='cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed',
        negative='drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly',
        image='styles/sai/photographic.jpg',
    ),
    StyleSchema(
        name='SAI Pixel Art',
        positive='pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics',
        negative='sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic',
        image='styles/sai/pixel_art.jpg',
    ),
    StyleSchema(
        name='SAI Texture',
        positive='texture {prompt} top down close-up',
        negative='ugly, deformed, noisy, blurry',
        image='styles/sai/texture.jpg',
    ),
]
