import cv2
import pytesseract
import base64
from PIL import Image
from io import BytesIO
import difflib
import numpy as np


def base64_to_cv2_image(base64_string):
    image_data = base64.b64decode(base64_string)
    pil_image = Image.open(BytesIO(image_data)).convert('RGB')
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv_image


def extract_text_from_base64(base64_string):
    image = base64_to_cv2_image(base64_string)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text.strip()


def number_to_words():
    return {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "@": "a"
    }


def words_to_number():
    return {v: k for k, v in number_to_words().items()}


def normalize_text(text):
    num_dict = words_to_number()
    word_dict = number_to_words()
    text = text.lower().strip().replace(',', '')
    text = ''.join(filter(str.isalnum, text))  # Loại ký tự lạ

    if text.isdigit():
        return text

    close_word = difflib.get_close_matches(text, num_dict.keys(), n=1, cutoff=0.6)
    if close_word:
        return num_dict[close_word[0]]

    return text


def sort_images_by_main(data):
    main_base64 = data.get("main")
    images = data.get("images", {})

    main_text = extract_text_from_base64(main_base64)
    words_order = [normalize_text(w) for w in main_text.replace(',', '').split()]

    image_text_map = {}

    for key, base64_str in images.items():
        text = extract_text_from_base64(base64_str)
        normalized_text = normalize_text(text)
        image_text_map[key] = normalized_text
        print(f"{key}: raw='{text}', normalized='{normalized_text}'")

    sorted_images = sorted(
        image_text_map.items(),
        key=lambda x: words_order.index(x[1]) if x[1] in words_order else float('inf')
    )

    print("\nMain Image Text:", main_text)
    result_keys = [key for key, _ in sorted_images]
    
    t = ("+".join(str(k) for k in result_keys))
    print(t)
    return sorted_images


# ====== SAMPLE USAGE =======

image_data = {
    "main": "iVBORw0KGgoAAAANSUhEUgAAAEcAAAAYCAYAAACoaOA9AAAACXBIWXMAAA7EAAAOxAGVKw4bAAADJUlEQVRYhe2XTUiUQRjHf4ZEhw6ySXioDh2sg9BhOjgVaSAVNXmoDl3EwNQOfUAfhERERHSIEPQUIiaUQkmGDB4qoiAco8aTiARFhIGIRHiQCJEOM+u+a+6+H7sEwf4vO/Mwz/+Z/c8zzzwvlFBCCUVGWXCipDgL3IzgN6KNbS/GBpQU/cBBYBlo0sa+TshTDzwG1oUsndbGHojCuZpoa8S9bI64Li+UFG04YQBmgfEC6KoIFya9LhJWk32P6GejBsgFJUU10BEwdWhjFwugnMNlXxi+RiUsC1ugpFgH9AKHvUkDrdok10dJUQ68BHZ6U9GuaY54bcAtP72sjR2I4lceQloO9AEN3vQGaC9EGI8bZIT5CVwvlDAXlBQALYFYQ1F9c4qjpFgP9AP13jQONGtjo6RuTigp9gFnAqYK4K2SYhJ4CgwVQfwgjgDb/HhQG/s7qmO+AnaJjDAAKeCKkiJxMfaCd64RNwXsB7qBYSXFxqQx1kCr/10GHsZxzCfOplXzauA87pRr4wQJ4BywJWRNLXA/IX8WlBQ1ng/glTb2Wxz/fOIM4jbZiVN82tsrgG6fBXHRFBiPAHW49mEXcJfMa6MKydAAgkW+J65zzpqjjZ0AJoI2JUUPoHCnvwdXoCNBSbGBTI8xS3ZhnwO6lBQ7gOO4Q6sBEjWEPl4l0Oinn7Sx7+JyRGmagvgcGKdi+v4ivA8J7qegwg+cBtLZ3ZeEYCVzlBRVwHtP+BG4A0wBCzghGsg8iRCjmQLQxqKkmAB24zLogZLinudJAafInPQyMBn733j4FqTZTxeAJ0l4gtdqFpgBtuP+wHAevyl/7VBSXACu4U79qjb2UR6/blx7AE6Ixhzrnmlj5xPwp3ESqPTjoaSd90oa+/vfghMpH2bIPI8ARz3PIjCaz1Eb+4LswrsWxnBixOYPIPh890b0+QtZNUcbOw3sxX2ZjwHzwBIuNSeA20CdNvYLrHSf1d59QBv7IyygNrYLOAY8xx3EEq5zHQMuAifSJ52EX0mRwmU/wGh6r0kQ+m0VspEq4ANOxEPa2LlC+P41fwkllFBCCf8J/gAQgun9EhHkfAAAAABJRU5ErkJggg==",
    "images": {
        "112": "iVBORw0KGgoAAAANSUhEUgAAAD4AAAAoCAYAAACmTknCAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABc0lEQVRoge3Vr0sEQRjG8a8icuHCYDIaVCZdEIPIgDYRs2iyGRQMNn8E0wWzcJgt2vwbhMFoMMigBqOYJhwGg1jGZU/2bne9vbs97v202Xl2doZ3eQeEEEIIIYQotbFuFzBKHwHrwIf1brP7LfXH+KA3MCgTWYNG6QugBjxY7w57t6X+GNmKj+zBU5ubUXof2EqYurbeXRqlT4C1hHkfMjdhnQpwDCwBlVhuxXqHUboKnAKLwGRs/g04t949hXUy5dL0suIK2DNKz4bxNrBK66HjdoBlWg8DMAOc/SPXUWpzs941gEaG5hZdZ0bpKeA2PJ8HXoG5MH623u0mvP87/2i9OwjrGKAOTBulq9a7Zo5cR0VW/DvlG6XqJ6XaTD9lvsczaFfxvGpG6bsCc4mk4iUSNa2CcomKqPhQ/jVlrHgkdk0BbLS7prLm4srU3F6ABZKb1ifQzJnrqEy/6RVwD3z9ef4O1K13eXNCCCGEEEKIIfQD1aN7QmEjpooAAAAASUVORK5CYII=",
        "114": "iVBORw0KGgoAAAANSUhEUgAAACkAAAAoCAYAAABjPNNTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAACkklEQVRYhe2WP2gUQRTGf4YQjiHFMQSMf7nCQsIRHAkYJaUKgmJpIbGx0thFCCJWIipikdJCBEFb/0QIRERQCf7DEZEgQeQMIYjFECUMIRzBYmbJy5qwe9xFLPZr9s17u99+O+/ttwsFChQoUKBAgYhNrSZURmvgFLAI3PXW+WY525tW9TcuA70xLgPXmiVsa5ZgDWwV8Y5WEG6EyDtAHfDAvQ3gbw2U0SVldEer+BqeSWV0J6Gl3UDdWzcpaueALcAvYAp48s9Fxt15AJRi6i0wKWrHRe1nzI/G3G/gqbduQhldBS4AS8BQlgM0upMHUmvZ0r1CIIl4YBnoibEHJoDdwM6YqxIedl00+uIcAp6JdVnE/SJ23rrpGH8X+e3x+Bi4HeN61k1zi1RGV4Au4INIV+KMAuwX+Tcink2L9NYtEcx+NsW3Jhpp92ngEVBL5fcoo2dY7Y+TIpYiVfwidQGDwLC3LvPGmSIj6SWgD3gOfCO0KLm2LyWwDrwX65kU5UHgJHDVW/clUyH52n02CgE44q2rA3Oivo3Q6uQlmvLWLYj6HKvn7gxw01v3Ko/AvCL3EaxiHhiLuXlRLwNHWdnNd0lBGQ1wntUuMOate5FXIOSbyXZCu19765aV0SVgl6gn9lKLx44osAKMAp3AtLhmsyRXRivgsLfu4XoC8uzkFHACKEfCi4CKtUVgCPgqzh9URo8D44R5PAa8FPWqMrotCuwFrpDhk5n/k8roLmAYGCAYc/JgM8B1b90nZfQAMEJo/UIU+DFpqzK6H7ghaGsEY68DI6kZblykEFsifCU6gHlv3ewa53QDP9K2EmfzFiujsUj4W7qfx4Ja/me+HmKLewgP+TkaeoECBQoUKPAf4Q80TKzWWDxdgQAAAABJRU5ErkJggg==",
        "113": "iVBORw0KGgoAAAANSUhEUgAAAFUAAAAoCAYAAACPSbZFAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAE5klEQVRoge2YbWiWVRjHf0mIDBEZI8zGQWyNmoeO+WHZy4dIyXBhEqK9EL1IDKEIqRgSfhCxkIIiCxt9SJKosV5AmRStFSNEQmzHDhJRUqc11hgSQ0TGGH24zt1zvL2f7X6kvYTn/+U593Nd9znnvs7/f13XfUNCQkJCQkJCQkLC/MM1c72BuYBRejGggeuAPkABTwBrgHrgPHASeMd6d7bW+a/WoB4E1ofLASTA1xa4XgAes965WuYvmmjewijdAjQDF613XxTYbwY2A7cAi4Bh4BvgiPVuMnIdi8arw/V+JMCNQAewEqgD9gEP1rLPecFUo/QC5GEagNPWu4kCnzVAV7gctd7dEdkA9gIPV1niZ6DdejcY/J8Fno/s2613/dF8K4EeKqRbZ73zZZ9nQVnHGcZrwNdI0F6t4nN/NO7L2eKA9gPtSI48Ef5rBg4bpevC9VB074U4oAAhj8aSX1viGf7FjMq/Bjl+DmwK481G6S7r3cncdOui8ZfRGppKQAeAZ7K5jdLfB1+FKOFp4G1gMJprvMr2f0JSA0gqKI0ZYapRGqP0XuAosB24E6msGxFWHjVKN2b+1rvvgDhH7gmSzuZrQgIDUpmPR74PROPu+LBCGumJ7BvCb8zUJUbpInL9EY1vKLBXRWmmhjZkI3Ab0naMAT8grDufc8/L8UPgIrADkVImxzbr3YXonruBxcH+JHAo2LJKDdCfy7k3RePbjdIGWI4wczmwMLJnjBsCJhFSLQh++ZwZs7mRGlCKqUbp9Yhs9wFbgHsRWe8BvgoSzHyL5NhnvTsOPBVtPpMjANa7EeBAtOxzRumGMC6UfgE2AVuRw1mBBHQUOA0cA94Ia00iqSjD8oK5Yjb/t0E1Sq9GHnYpcAZh1EvRog3ZZgOuRI4ZDiGVGmAJ8HII7K3hv3GE+TF+j8YDwG7k8DYAq0KX0IGkooHId7qgxUxdapReVOBTiDLy3xH57bHenQIwSg8g1XqCS6VzJXIEhEFG6d1UWqfsgLLDP1GQavqBx8N4ofXu49holN6KKAzkwNrCOA7qZUy13o0apcej/S4Dfsv7FaFMUONT7AwV9U/gHMLak9a74cI7KxU9xijyQIPAj3mj9e6UUfoz4KHwV8z8Iul/iwSrGWgxSr8IvGm9mwgFriPy7YrGcSG6vsr+DwTbr1yec6uiTFCzDYOkgPty9kmj9DHghSD1vBw/RQI4BAxa78bDw64A/q6y5n6kOC2J1wF6847WO4zSO4GPgn878IhR+hzSMWQsf99690F0ay9wI/AXcKRoE9a7d6vsb0pM+0YVCk830x9Ap/XudaP0PcB74b8z1rtLXvHycrTetVEAo/SjSCHMcMp6t22KfTYCu5Aimu11EvkwcjC0bbOCUq+poVjtBFqpHtwR691dob/socLuTi6VYxcVBu7NsSe/bjeVBnyX9e6TEnutQ1LWJOCtd9Wa+xlD2aC2IFI6i/Sb9chns1bkdRBg3Hq3Kvg3U5EjSE9bJMdXpll3MRKgIevd2FS+8wll5L+Fyvv4MLDNejcUbGuBw8HWa73bEd03b+Q42ygT1GWInGPWOaTFyFqiQSTYIwX3z7kcZxtl5d+EFI3WnGkMqe5vFfSPVy1q+p5qlK4HmhA5jwC/WF/TR/GEhISEhISEhISEhISEhISE/z3+AcSyrI77AVZUAAAAAElFTkSuQmCC"
    }
}

sorted_images = sort_images_by_main(image_data)



