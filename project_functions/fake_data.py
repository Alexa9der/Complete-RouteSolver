from lib.imports import *

def data_generation(size = 100, random_seed = None):
    """
    Generate synthetic data for waste collection scheduling.

    Parameters:
    - size (int): Number of rows/data points to generate. Default is 100.
    - random_seed (int): Seed for random number generation. Default is None.

    Returns:
    - pd.DataFrame: A DataFrame containing synthetic data for waste collection scheduling.

    Example:
    >>> df = dat_generation(size=50, random_seed=42)
    """
    # Set up Faker instance
    fake = Faker('pl_PL')  
    fake.random.seed(random_seed)

    # Define lists for periodicity, customer types, waste codes, and container types
    cities = ['Ołtarzew', 'Siestrzeń', 'Gołków', 'Kozerki', 'Raszyn', 'Łazy',
           'Młochów', 'Grodzisk Mazowiecki', 'Warszawa', 'Żyrardów',
           'Piaseczno', 'Kajetany', 'Koszajec', 'Pęcice', 'Radziejowice',
           'Słubica A', 'Janinów', 'Grabce-Towarzystwo', 'Baranów', 'Błonie',
           'Podkowa Leśna', 'Rozalin', 'Radziejowice Parcel', 'Parole',
           'Pruszków', 'Nowa Wola', 'Rybie', 'Książenice', 'Siedliska',
           'Huta Żabiowolska', 'Konstancin-Jeziorna', 'Jawczyce', 'Marynin',
           'Radonice', 'Wola Przypkowska', 'Puchały', 'Brwinów', 'Słabomierz',
           'Szamoty', 'Wręcza', 'Kotowice', 'Domaniewek', 'Nowa Wieś',
           'Adamowizna', 'Michałowice', 'Milanówek', 'Piastów', 'Mszczonów',
           'Walendów', 'Henryków-Urocze', 'Wólka Kosowska', 'Nadarzyn',
           'Kozery', 'Stara Iwiczna', 'Janczewice', 'Kałęczyn', 'Jaktorów',
           'Parzniew', 'Janki', 'Sokołów', 'Nowa Iwiczna', 'Biskupice',
           'Pilaszków', 'Kuklówka Zarzeczna', 'Głosków', 'Wypędy',
           'Żabia Wola', 'Święcice', 'Mysiadło', 'Rusiec',
           'Ożarów Mazowiecki', 'Kolonia Lesznowola', 'Wola Gołkowska',
           'Wolica', 'Wilcza Góra', 'Stare Budy', 'Sękocin Nowy',
           'Odrano-Wola', 'Wiskitki', 'Stara Wieś', 'Julianów',
           'Zalesie Górne', 'Guzów', 'Strzeniówka', 'Otrębusy', 'Józefosław',
           'Kostowiec', 'Antoninów', 'Błonie-Wieś', 'Natolin',
           'Nowy Drzewicz', 'Lesznowola', 'Lisówek', 'Jabłonowo',
           'Moszna Parcela', 'Urzut', 'Żabieniec', 'Sękocin Stary',
           'Nowa Bukówka', 'Baniocha', 'Kłudzienko', 'Kolonia Warszawska',
           'Szymanów', 'Mieszkowo', 'Chylice Kolonia', 'Kuleszówka',
           'Kłudno Stare', 'Jaktorów Kolonia', 'Podolszyn Nowy', 'Bronisze',
           'Komorów', 'Urszulin', 'Kopytów', 'Macierzysz', 'Chrzanów Duży',
           'Adamów', 'Jazgarzewszczyzna', 'Stara Bukówka', 'Konotopa',
           'Kanie', 'Michałowice Wieś', 'Stare Babice', 'Bogatki', 'Jaworowa',
           'Opacz', 'Badowo-Mściska', 'Jastrzębnik', 'Falenty Nowe',
           'Rumianka', 'Wólka Pracka', 'Kady', 'Dawidy', 'Chylice-Kolonia',
           'Mory', 'Stary Drzewicz', 'Makówka', 'Świętochów', 'Duchnice',
           'Runów', 'Grzędy', 'Tarczyn', 'Bartoszówka', 'Suchodół',
           'Podolszyn', 'Przeszkoda', 'Reguły', 'Owczarnia', 'Koprki',
           'Bielawa', 'Wólka Kozodawska', 'Jesówka', 'Leszno', 'Chylice',
           'Mroków', 'Chrzanów Mały', 'Wola Mrokowska', 'Władysławów',
           'Słubica B', 'Orzeszyn', 'Międzyborów', 'Piotrkówek Mały',
           'Falenty', 'Warszawianka', 'Pawłowice', 'Solec', 'Tartak Brzózki',
           'Złotokłos', 'Mościska', ' Wola Gołkowska', 'Opypy', 'Płochocin',
           'Michałowice Osiedle', 'Henryszew', 'Żółwin', 'Czarny Las',
           'Laszczki', 'Magdalenka', 'Żuków', 'Raszyn-Rybie', 'Łoziska',
           'Moszna Wieś', 'Dawidy Bankowe', 'Blizne Łaszczyńskiego',
           'Jazgarzew', 'Cyganka', 'Tłuste', 'Kosów', 'Nowe Grocholice',
           'Kaleń', 'Wymysłów', 'Żelechów', 'Bieniewo-Parcela', 'Bieniewiec',
           'Płochocin-Osiedle', 'Kamionka', 'Opacz Kolonia', 'Tłumy',
           'Józefina', 'Osowiec', 'Robercin', 'Błonie-Pass', 'Korytów A',
           'Lubiczów', 'Kotorydz', 'Krze Duże', 'Stefanowo', 'Terenia',
           'Bąkówka', 'Krzyżówka', 'Bobrowiec', 'Łady', 'Budy Mszczonowskie',
           'Sękocin Las', 'Chlebnia', 'Pęcice Małe', 'Blizne Jasińskiego',
           'Krakowiany', 'Wycinki Osowskie', 'Komorniki', 'Musuły',
           'Wólka Grodziska', 'Daniewice', 'Kuklówka Radziejowicka',
           'Babice Nowe', 'Grzymek', 'Głosków-Letnisko', 'Wola Krakowiańska',
           'Czubin', 'Warszawa- Targówek', 'Przypki', 'Budy Zosine',
           'Falenty Duże']
    periodicity = ['2 wtorek', '2 i 4 piątek', '1 i 3 środa', '1 poniedziałek',
               '4 czwartek', 'co tydzień poniedziałek', 'co tydzień czwartek',
               '1 i 3 wtorek', ' - - - - - - -', '1 i 3 piątek', '2 i 4 czwartek',
               '1 i 3 poniedziałek', '2 i 4 poniedziałek', '2 i 4 środa',
               'co tydzień piątek', 'poniedziałek czwartek sobota',
               '1 i 3 czwartek', 'wtorek piątek', 'poniedziałek czwartek',
               'co tydzień środa', 'co tydzień wtorek', 'Wtorek co 3 tygodnie',
               '4 środa', '4 piątek', '3 poniedziałek',
               'poniedziałek wtorek środa czwartek piątek',
               'poniedziałek środa piątek', '2 środa', '1 piątek', '1 środa',
               '3 czwartek', '1 czwartek', '2 czwartek', '3 wtorek',
               '2 i 4 wtorek', '3 środa', ' - W - - - - - 4',
               'poniedziałek środa', '3 piątek', 'środa piątek',
               'wtorek czwartek sobota', 'wtorek czwartek', '4 poniedziałek',
               '2 poniedziałek', '2 piątek', '4 wtorek', 'poniedziałek piątek',
               '4 wtorek co 4 tygodnie', 'poniedziałek wtorek czwartek piątek',
               ' P - - C - - - 1 2 3 4 5 O',
               'poniedziałek wtorek środa piątek sobota', 'środa sobota',
               '1 wtorek', '3 piątek listopad marzec',
               'co tydzień piątek kwiecień październik', '6 dni w tygodniu',
               '2 wtorek co 4 tygodnie', '2 i 4 środa kwiecień październik',
               'Ostatnia środa', 'co tydzień sobota',
               'poniedziałek raz na 3 miesiące',
               'co tydzień środa kwiecień październik', ' - - - C - - - 4',
               ' - - - - P - - co 1', 'Co 4 tygodnie piątek',
               '2 piątek co 4 tygodnie', 'wtorek sobota',
               'co tydzień poniedziałek kwiecień październik',
               ' - - - - - - - co 0', ' - - Ś - - - - 4', ' P - - - - - -',
               '3 czwartek listopad marzec',
               'co tydzień czwartek kwiecień październik', ' - - Ś - - - - co 2',
               'Czwartek co 4 tygodnie II',
               '2 i 4 czwartek kwiecień  październik', ' P - - - - - - 4',
               ' P - - - - - - 2', ' - - - C - - - co 1', ' - - - - P - - 4',
               '2 i 4 sobota', '2 i 4 piątek kwiecień październik',
               '2 i 4 poniedziałek kwiecień październik',
               '3 poniedziałek listopad-marzec', 'ostatni czwartek',
               'co 4 tygodnie wtorek', 'czwartek sobota', ' - - Ś - - - - co 1',
               '1 i 3 sobota']
    customer_types = ['BWA', 'BRA', 'BHC', 'BRC', 'BWC', 'BRB', 'BWB', 'BHB', 'BHA',
                       'PH', 'BRAB', 'PW', 'PR']
    waste_codes = [70681, 200301, 150102, 150106, 150101, 150107, 200108,
                  200201, 200139, 200102, 200101, 150104,  40209]
    type_containers = ['L-120K', 'L-1100K', 'L-240K', 'UP-7K', 'L-700K', 'L-770K',
                       'UP-5.K', 'L-660K', 'UP-2,5K', 'L-240T', 'L-1100T', 'L-120T',
                       'UP-5.T', 'UP-7T', 'L-770T', 'L-660T', 'UP-7.T', 'UP-2,5T',
                       'L-240P', 'L-1100P', 'L-120P', 'UP-5.P', 'UP-7P', 'L-660P',
                       'L-770P', 'UP-2,5P', 'L-120S', 'L-240S', 'L-1100S', 'L-660S',
                       'L-770S', 'L-240B', 'L-120B', 'L-770B', 'L-1100B', 'L-660B']
    
    # Create a list to store generated data
    data_for_df = []

    # # Generate synthetic data
    for _ in range(size): 
        customer_type = fake.random_element(customer_types)
        city = fake.random_element(cities) 
        
        # zip_code = fake.zipcode()
        street = fake.street_name()
        nr_budynku = fake.building_number()
        frequency = fake.random_element(periodicity)
        waste_code = fake.random_element(waste_codes)
        type_container = fake.random_element(type_containers)
        
    
        data_for_df.append([customer_type, city, street, # zip_code
                            nr_budynku, frequency, waste_code, type_container])
    
    # Create a dataframe
    df = pd.DataFrame(data_for_df, columns=['Rodzaj klienta', 'Miasto', # 'Kod pocz.',
                                             'Ulica', 'Nr budynku',"Cykl odbiorów", 
                                            "Kod odpadu", "Typ pojemnika"])
    return df 
