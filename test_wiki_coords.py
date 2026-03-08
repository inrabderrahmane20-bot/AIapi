
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='CityExplorer/3.0 (https://traveltto.com)'
)

def check_coords(city_name):
    page = wiki.page(city_name)
    if page.exists():
        print(f"Page '{city_name}' exists.")
        try:
            # Check if coordinates property exists (it might not in all versions/wrappers)
            if hasattr(page, 'coordinates'):
                print(f"Coordinates: {page.coordinates}")
            else:
                print("No 'coordinates' attribute on page object.")
        except Exception as e:
            print(f"Error accessing coordinates: {e}")
    else:
        print(f"Page '{city_name}' does not exist.")

check_coords("Rabat")
check_coords("Paris")
