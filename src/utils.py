COLUMNS_CalculateWeightAndTrimAction = {
    "DEADLOAD_MAC": "DEADLOAD_MAC",
    "DEZYLOZY_MAC": "DEADLOAD_MAC",
    "ESTIMATED_TRAFFIC_LOAD": "ESTIMATED_TRAFFIC_LOAD",
    "ESTIMATED_TRAFFIC_LOZY": "ESTIMATED_TRAFFIC_LOAD",
    "IDEAL_ADDITIONAL_LOAD_AFT": "IDEAL_ADDITIONAL_LOAD_AFT",
    "IDEAL_ADDITIONAL_LOAD_FWD": "IDEAL_ADDITIONAL_LOAD_FWD",
    "IDEAL_ZYDITIONAL_LOZY_AFT": "IDEAL_ADDITIONAL_LOAD_AFT",
    "IDEAL_ZYDITIONAL_LOZY_FWD": "IDEAL_ADDITIONAL_LOAD_FWD",
    "LOAD_TO_AFT": "LOAD_TO_AFT",
    "LOAD_TO_FWD": "LOAD_TO_FWD",
    "LOZY_TO_AFT": "LOAD_TO_AFT",
    "LOZY_TO_FWD": "LOAD_TO_FWD",
    "TOTAL_DEADLOAD_WI_index": "TOTAL_DEADLOAD_WI_index",
    "TOTAL_DEADLOAD_WI_weight": "TOTAL_DEADLOAD_WI_weight",
    "TOTAL_DEZYLOZY_WI_index": "TOTAL_DEADLOAD_WI_index",
    "TOTAL_DEZYLOZY_WI_weight": "TOTAL_DEADLOAD_WI_weight",
    "TOTAL_LOAD_WI": "TOTAL_LOAD_WI",
    "TOTAL_LOZY_WI": "TOTAL_LOAD_WI",
    "TOTAL_TRAFFIC_LOAD": "TOTAL_TRAFFIC_LOAD",
    "TOTAL_TRAFFIC_LOZY": "TOTAL_TRAFFIC_LOAD",
    "UNDERLOAD": "UNDERLOAD",
    "UNDERLOZY": "UNDERLOAD",
    "LIMITING_WEIGHT": "LIMITING_WEIGHT",
    "LIMITING_WMNGHT": "LIMITING_WEIGHT",
}

COLUMNS_AssignLCCAction = [
    "A/L",
    "Flt No",
    "Suff",
    "Date",
    "DEP",
    "ARR",
    "STD in UTC",
    "ETD in UTC",
    "AC Type",
    "Reg",
    "Assigned user",
    "Planned user",
    "Location",
]

COLUMNS_UpdateFlightAction_METADATA = ["Airline", "Flightnumber", "Suffix", "Date"]

COLUMNS_UpdateCrewDataAction = [
    "Cockpit crew nr.",
    "Cockpit crew bag nr.",
    "Cabin crew nr.",
    "Cabin crew bag nr.",
    "As crew nr.",
    "As pax nr.",
    "Deadhead cabin crew nr.",
    "Deadhead cockpit crew nr.",
    "Captain`s name",
]

COLUMNS_UpdateFlightAction_RECEIVED = [
    "DEP",
    "STD in UTC",
    "ETD in UTC",
    "DEST",
    "TRANSIT",
    "CANCELLED",
    "DELETED",
]

COLUMNS_UpdateFlightAction_SAVED = [
    "DEP",
    "STD in UTC",
    "ETD in UTC",
    "DEST",
    "TRANSIT",
    "CANCELLED",
]

COLUMNS_StoreRegistrationAndConfigurationAc = [
    "Start Weight",
    "Start Index",
    "Crew",
    "Water(%)",
    "Total Weight",
    "Index",
]

COLUMNS_StoreRegistrationAndConfigurationAc_STATUS_KEYS = [
    "AIRBORNE",
    "AIRCRAFT_CONFIG",
    "AUTOMATION_STARTED",
    "AUTO_MODE_ACTIVE",
    "BAG_LOAD_ITEMS_GEN",
    "CABIN_CONFIG",
    "CARGO_FINAL",
    "CARGO_TRANSFER",
    "CHECK_IN_FINAL",
    "DGR_ITEMS",
    "EZFW",
    "EZFW_COUNTER",
    "FUEL",
    "FUEL_ORDER",
    "LOADING_INSTRUCTION",
    "OFFBLOCK",
    "OFP",
    "REGISTRATION",
    "REGISTRATION_CHANGE",
]

COLUMNS_UpdateLoadTableAction = [
    "Total baggage",
    "Total cargo",
    "Total EIC",
    "Total mail",
]

COLUMNS_UpdateLoadTableAction_STATUS_KEYS = [
    "AIRBORNE",
    "AIRCRAFT_CONFIG",
    "ALLOWANCE_CHECK_PERFORMED",
    "AUTOMATION_STARTED",
    "AUTO_MODE_ACTIVE",
    "BAG_LOAD_ITEMS_GEN",
    "BAG_LOZY_ITEMS_GEN",
    "BAG_ULD_ORD",
    "CABIN_CONFIG",
    "CALC_HIST_DATA",
    "CARGO_FINAL",
    "CARGO_TRANSFER",
    "CHECK_IN_FINAL",
    "CHECK_IN_OPEN",
    "DGR_ITEMS",
    "EZFW",
    "EZFW_COUNTER",
    "FUEL",
    "FUEL_ORDER",
    "LDM",
    "LOADING_INSTRUCTION",
    "LOADSHEET",
    "LOZYING_INSTRUCTION",
    "LOZYSHEET",
    "OFFBLOCK",
    "OFP",
    "PDM",
    "REGISTRATION",
    "REGISTRATION_CHANGE",
    "TRANSIT_ACCEPTANCE",
    "TRANSIT_PAX",
]

COLUMNS_StorePaxDataAction_saved = [
    "TOTAL Pax",
    "Y",
    "Jump",
    "StandBy",
    "Male",
    "Female",
    "Child",
    "Infant",
    "Total bag",
    "Total bag weight",
    "Baggage weight type",
]

COLUMNS_StorePaxDataAction_STATUS_KEYS_saved = [
    "AIRBORNE",
    "AIRCRAFT_CONFIG",
    "ALLOWANCE_CHECK_PERFORMED",
    "AUTOMATION_STARTED",
    "AUTO_MODE_ACTIVE",
    "BAG_LOAD_ITEMS_GEN",
    "BAG_LOZY_ITEMS_GEN",
    "BAG_ULD_ORD",
    "CABIN_CONFIG",
    "CALC_HIST_DATA",
    "CARGO_FINAL",
    "CARGO_TRANSFER",
    "CHECK_IN_FINAL",
    "CHECK_IN_OPEN",
    "CPM",
    "DGR_ITEMS",
    "EZFW",
    "EZFW_COUNTER",
    "FINAL_RELEASE",
    "FUEL",
    "FUEL_ORDER",
    "FUEL_STATUS_FINAL",
    "LDM",
    "LOADING_INSTRUCTION",
    "LOADSHEET",
    "OFFBLOCK",
    "OFP",
    "PDM",
    "RAMP_FINAL",
    "REGISTRATION",
    "REGISTRATION_CHANGE",
    "TRANSIT_ACCEPTANCE",
    "TRANSIT_PAX",
    "UCM",
]

COLUMNS_FuelDataInitializer_STATUS_KEYS = [
    "AB",
    "AC",
    "Am",
    "As",
    "BI",
    "CC",
    "CF",
    "CKo",
    "CT",
    "EZ",
    "Ec",
    "FR",
    "HD",
    "OB",
    "R",
    "RC",
    "RF",
]

COLUMNS_FuelDataInitializer = ["trip", "taxi", "takeoff", "ballast", "edno"]

COLUMNS_UpdateFuelDataAction_ANONYMIZATION = {
    "BAG_LOAD_ITEMS_GEN": "BAG_LOZY_ITEMS_GEN",
    "LOADSHEET": "LOZYSHEET",
    "LOADING_INSTRUCTION": "LOZYING_INSTRUCTION",
}

COLUMNS_UpdateFuelDataAction_STATUS_KEYS = [
    "AIRBORNE",
    "AIRCRAFT_CONFIG",
    "ALLOWANCE_CHECK_PERFORMED",
    "AUTOMATION_STARTED",
    "AUTO_MODE_ACTIVE",
    "BAG_LOAD_ITEMS_GEN",
    # "BAG_LOZY_ITEMS_GEN",
    "BAG_ULD_ORD",
    "CABIN_CONFIG",
    "CALC_HIST_DATA",
    "CARGO_FINAL",
    "CARGO_TRANSFER",
    "CHECK_IN_FINAL",
    "DGR_ITEMS",
    "EZFW",
    "EZFW_COUNTER",
    "FINAL_RELEASE",
    "FUEL",
    "FUEL_ORDER",
    "LOADING_INSTRUCTION",
    "LOADSHEET",
    # "LOZYING_INSTRUCTION",
    # "LOZYSHEET",
    "OFFBLOCK",
    "OFP",
    "RAMP_FINAL",
    "REGISTRATION",
    "REGISTRATION_CHANGE",
    "TRANSIT_ACCEPTANCE",
    "TRANSIT_PAX",
]

COLUMNS_UpdateFuelDataAction_FUEL_KEYS = [
    "CT",
    "LI",
    "LM",
    "LO",
    "M1",
    "RI",
    "RM",
    "RO",
    "TRIM",
    "ballast",
    "edno",
    "takeoff",
    "taxi",
    "trip",
]

COLUMNS_UpdateFuelDataAction_MW_KEYS = ["mzfw", "mtxw", "mtow", "mlaw"]

COLUMNS_UpdateFuelDataAction_received = [
    "StandardFuel",
    "Non standard fuel",
    "EditionNumber",
    "Fueling Indicator",
    "TruckOnSbyIndicator",
    "Manual Non Standard Index",
    "Actual OFP No",
    "Density",
    "Trip Fuel",
    "FZFW",
    "Max Fuel Cap",
    "EZFW sent",
    "Minimum TOF",
    "Triggerd by",
    "Take Off Fuel",
    "Taxi Fuel",
    "Planning Fuel",
    "Flight Time",
    "Remarks",
]

airport_coords = {
    "BLR": [12.972442, 77.580643],  # Bangalore, India
    "BOM": [19.089560, 72.865614],  # Mumbai, India
    "DEL": [28.556162, 77.100281],  # Delhi, India
    "GOX": [15.380055, 73.832678],  # Goa, India
    "IXB": [26.681206, 88.328617],  # Bagdogra, India
    "COK": [10.152000, 76.401905],  # Cochin, India
    "CCU": [22.654722, 88.446722],  # Kolkata, India
    "HYD": [17.240263, 78.429385],  # Hyderabad, India
    "GAU": [26.106092, 91.585453],  # Guwahati, India
    "SXR": [33.987140, 74.774343],  # Srinagar, India
    "AMD": [23.072059, 72.631652],  # Ahmedabad, India
    "VNS": [25.452362, 82.861406],  # Varanasi, India
    "PNQ": [18.580208, 73.919353],  # Pune, India
    "LKO": [26.760594, 80.889573],  # Lucknow, India
    "AYJ": [24.086592, 85.551064],  # Aizawl, India
    "IXZ": [11.641035, 92.729370],  # Port Blair, India
    "GWL": [26.293686, 78.227564],  # Gwalior, India
    "BBI": [20.244444, 85.817778],  # Bhubaneswar, India
    "DOH": [25.273056, 51.608056],  # Doha, Qatar
    "IXA": [23.886978, 91.240450],  # Agartala, India
    "MAA": [12.990005, 80.169296],  # Chennai, India
    "SIN": [1.364420, 103.991531],  # Singapore
    "DXB": [25.252778, 55.364444],  # Dubai, UAE
    "RUP": [26.139186, 89.909905],  # Rupsi, India
    "DUB": [53.421389, -6.270000],  # Dublin, Ireland
    "LAX": [33.941589, -118.408530],  # Los Angeles, USA
    "LHR": [51.470020, -0.454295],  # London Heathrow, UK
    "IAD": [38.953116, -77.456539],  # Washington Dulles, USA
    "MAD": [40.472164, -3.560046],  # Madrid, Spain
    "MUC": [48.353783, 11.786086],  # Munich, Germany
    "MAN": [53.365004, -2.272406],  # Manchester, UK
    "AGP": [36.674856, -4.499047],  # Malaga, Spain
    "YYZ": [43.677717, -79.624820],  # Toronto, Canada
    "ORD": [41.974162, -87.907321],  # Chicago O'Hare, USA
    "JFK": [40.641311, -73.778139],  # New York JFK, USA
    "NOC": [53.910297, -8.818490],  # Knock, Ireland
    "SFO": [37.621313, -122.378955],  # San Francisco, USA
    "SNN": [52.701978, -8.924817],  # Shannon, Ireland
    "EWR": [40.689531, -74.174462],  # Newark, USA
    "MCO": [28.431157, -81.308083],  # Orlando, USA
    "BDL": [41.938874, -72.683228],  # Hartford, USA
    "BOS": [42.365613, -71.009560],  # Boston, USA
    "PHL": [39.874403, -75.242422],  # Philadelphia, USA
    "SEA": [47.450249, -122.308817],  # Seattle, USA
    "CLE": [41.412433, -81.847030],  # Cleveland, USA
    "VCE": [45.504928, 12.339932],  # Venice, Italy
    "CDG": [49.009690, 2.547925],  # Paris Charles de Gaulle, France
    "FAO": [37.017639, -7.968475],  # Faro, Portugal
    "BGI": [13.074603, -59.492456],  # Bridgetown, Barbados
    "MSP": [44.883055, -93.210573],  # Minneapolis, USA
    "PGF": [42.740071, 2.870187],  # Perpignan, France
    "NAP": [40.886033, 14.290781],  # Naples, Italy
    "BOD": [44.828335, -0.715556],  # Bordeaux, France
    "FCO": [41.799886, 12.246238],  # Rome Fiumicino, Italy
    "DUS": [51.289453, 6.766775],  # Dusseldorf, Germany
    "TLS": [43.629311, 1.364114],  # Toulouse, France
    "CMH": [39.997985, -82.890987],  # Columbus, USA
    "LPA": [27.933104, -15.386442],  # Las Palmas, Spain
    "BVB": [2.843611, -60.692222],  # Boa Vista, Brazil
    "SSA": [-12.910999, -38.331998],  # Salvador, Brazil
    "REC": [-8.125556, -34.923889],  # Recife, Brazil
    "MCZ": [-9.510808, -35.791660],  # Maceió, Brazil
    "SLZ": [-2.585361, -44.234139],  # São Luís, Brazil
    "NAT": [-5.769803, -35.376387],  # Natal, Brazil
    "STM": [-2.424722, -54.785833],  # Santarém, Brazil
    "CGB": [-15.652929, -56.117739],  # Cuiabá, Brazil
    "IMP": [-5.530833, -47.459722],  # Imperatriz, Brazil
    "PVH": [-8.709289, -63.902333],  # Porto Velho, Brazil
    "MAO": [-3.038611, -60.049721],  # Manaus, Brazil
    "CGR": [-20.469522, -54.672500],  # Campo Grande, Brazil
    "GRU": [-23.435556, -46.473056],  # São Paulo, Brazil
    "VIX": [-20.258056, -40.286389],  # Vitória, Brazil
    "THE": [-5.059944, -42.823661],  # Teresina, Brazil
    "MOC": [-16.707522, -43.818902],  # Montes Claros, Brazil
    "SJP": [-20.816597, -49.406506],  # São José do Rio Preto, Brazil
    "FOR": [-3.776283, -38.532556],  # Fortaleza, Brazil
    "GYN": [-16.626861, -49.226794],  # Goiânia, Brazil
    "UDI": [-18.883611, -48.225277],  # Uberlândia, Brazil
    "CWB": [-25.528475, -49.175784],  # Curitiba, Brazil
    "CNF": [-19.624444, -43.971944],  # Belo Horizonte, Brazil
    "POA": [-29.994428, -51.171428],  # Porto Alegre, Brazil
    "BSB": [-15.871111, -47.918611],  # Brasília, Brazil
    "BEL": [-1.384722, -48.477778],  # Belém, Brazil
    "GIG": [-22.809999, -43.250556],  # Rio de Janeiro, Brazil
    "PPB": [-22.175139, -51.424883],  # Presidente Prudente, Brazil
    "FLN": [-27.670278, -48.552500],  # Florianópolis, Brazil
    "IGU": [-25.598889, -54.487222],  # Foz do Iguaçu, Brazil
    "AJU": [-10.984000, -37.070333],  # Aracaju, Brazil
    "CGH": [-23.626667, -46.655556],  # São Paulo Congonhas, Brazil
    "NVT": [-26.880833, -48.651111],  # Navegantes, Brazil
    "VCP": [-23.007222, -47.134444],  # Campinas, Brazil
    "JDO": [-7.218056, -39.270556],  # Juazeiro do Norte, Brazil
    "SDU": [-22.910444, -43.163133],  # Rio de Janeiro Santos Dumont, Brazil
    "IPN": [-19.471110, -42.487779],  # Ipatinga, Brazil
    "ORY": [48.725278, 2.359167],  # Paris Orly, France
    "RAO": [-21.136667, -47.776944],  # Ribeirão Preto, Brazil
    "LIS": [38.774167, -9.134167],  # Lisbon, Portugal
    "FEN": [-3.854928, -32.423333],  # Fernando de Noronha, Brazil
    "FLL": [26.072556, -80.152778],  # Fort Lauderdale, USA
    "LDB": [-23.333616, -51.133553],  # Londrina, Brazil
    "UBA": [-19.764722, -47.964722],  # Uberaba, Brazil
    "CAC": [-25.000000, -53.500000],  # Cascavel, Brazil
    "PNZ": [-9.362411, -40.569097],  # Petrolina, Brazil
    "JPA": [-7.146389, -34.948611],  # João Pessoa, Brazil
    "BPS": [-16.438611, -39.080833],  # Porto Seguro, Brazil
    "IZA": [-21.513056, -43.173056],  # Zona da Mata, Brazil
    "ARU": [-21.141319, -50.424725],  # Araçatuba, Brazil
    "VDC": [-14.861944, -40.863056],  # Vitória da Conquista, Brazil
    "JOI": [-26.224444, -48.797223],  # Joinville, Brazil
    "JTC": [-22.157780, -49.068481],  # Bauru, Brazil
    "MGF": [-23.479444, -52.012222],  # Maringá, Brazil
    "PFB": [-28.243999, -52.327731],  # Passo Fundo, Brazil
    "CMG": [-19.011940, -57.673057],  # Corumbá, Brazil
    "CAW": [-21.698333, -41.301667],  # Campos, Brazil
    "URG": [-29.782778, -57.038889],  # Uruguaiana, Brazil
    "BYO": [-10.838889, -38.073611],  # Bayeux, Brazil
    "CKS": [-6.115833, -50.001111],  # Carajás, Brazil
    "BVH": [-12.694722, -60.098333],  # Vilhena, Brazil
    "LEC": [-12.482222, -41.283056],  # Lençóis, Brazil
    "OPS": [-11.885000, -55.586111],  # Sinop, Brazil
    "MAB": [-5.368611, -49.133333],  # Marabá, Brazil
    "SJL": [-0.148333, -66.985833],  # São Gabriel da Cachoeira, Brazil
    "MII": [-22.196944, -49.926944],  # Marília, Brazil
    "OAL": [-9.482778, -40.290278],  # Cacoal, Brazil
    "MCP": [0.050833, -51.072500],  # Macapá, Brazil
    "IOS": [-14.815000, -39.033333],  # Ilhéus, Brazil
    "EEA": [-27.094444, -52.637222],  # Erechim, Brazil
    "PET": [-31.718333, -52.327778],  # Pelotas, Brazil
    "XAP": [-27.134167, -52.660000],  # Chapecó, Brazil
    "CPV": [-7.270000, -35.896944],  # Campina Grande, Brazil
    "SMT": [-14.491111, -57.434167],  # Sorriso, Brazil
    "CXJ": [-29.197778, -51.187500],  # Caxias do Sul, Brazil
    "AAX": [-19.561667, -46.964444],  # Araxá, Brazil
    "BRA": [-12.083333, -45.000000],  # Barreiras, Brazil
    "FEC": [-12.201944, -38.948056],  # Feira de Santana, Brazil
    "ROO": [-17.553333, -56.383611],  # Rondonópolis, Brazil
    "GPB": [-25.383333, -51.466667],  # Guarapuava, Brazil
    "AFL": [-9.866667, -56.100000],  # Alta Floresta, Brazil
    "PMG": [-22.549444, -55.702778],  # Ponta Porã, Brazil
    "JJG": [-28.231389, -48.642500],  # Jaguaruna, Brazil
    "MVD": [-34.837600, -56.030800],  # Montevideo, Uruguay
    "ATM": [-3.253333, -52.254722],  # Altamira, Brazil
    "PAV": [-9.358333, -40.569722],  # Paulo Afonso, Brazil
    "PMW": [-10.241667, -48.352222],  # Palmas, Brazil
    "ITB": [-4.242222, -56.000556],  # Itaituba, Brazil
    "CFB": [-23.007500, -47.134444],  # Campinas, Brazil
    "TFF": [-3.382222, -64.724167],  # Tefé, Brazil
    "GEL": [-32.332500, -54.110833],  # Pelotas, Brazil
    "MDZ": [-32.831944, -68.792222],  # Mendoza, Argentina
    "POJ": [-18.672500, -46.491389],  # Patos de Minas, Brazil
    "PIN": [-2.646111, -56.736944],  # Parintins, Brazil
    "JPR": [-10.870556, -61.846389],  # Ji-Paraná, Brazil
    "RIA": [-29.210278, -53.677778],  # Santa Maria, Brazil
    "TBT": [-4.255667, -69.937500],  # Tabatinga, Brazil
    "CUR": [12.188056, -68.959167],  # Curaçao
    "CLV": [-17.725278, -48.607778],  # Caldas Novas, Brazil
    "TJL": [-23.350278, -49.369167],  # Toledo, Brazil
    "PDP": [-34.914444, -54.920000],  # Punta del Este, Uruguay
    "SSV": [-0.982222, -48.629167],  # São Salvador, Brazil
    "GNM": [-10.234444, -40.477500],  # Guanambi, Brazil
    "UNA": [-15.355833, -38.994444],  # Una, Brazil
    "JJD": [-2.907222, -40.351944],  # Jericoacoara, Brazil
    "PHB": [-2.893333, -41.732778],  # Parnaíba, Brazil
    "TMT": [-1.489444, -56.396111],  # Trombetas, Brazil
    "GVR": [-18.895000, -41.982778],  # Governador Valadares, Brazil
    "MVF": [-5.201667, -37.380556],  # Mossoró, Brazil
    "RVD": [-17.720556, -49.167500],  # Rio Verde, Brazil
    "PGZ": [-25.528333, -54.635833],  # Ponta Grossa, Brazil
    "PTO": [-25.595000, -49.424722],  # Pato Branco, Brazil
    "RBB": [-9.583333, -67.633333],  # Borba, Brazil
    "MEU": [0.889722, -52.602222],  # Macapá, Brazil
    "TFL": [-17.892500, -41.506944],  # Teófilo Otoni, Brazil
    "VAG": [-21.555278, -45.473333],  # Varginha, Brazil
    "LHN": [-9.237500, -56.490556],  # Laranjal, Brazil
    "MNX": [-5.811944, -61.299444],  # Manicoré, Brazil
    "ERN": [-6.640556, -69.879444],  # Eirunepé, Brazil
    "ARX": [-6.235833, -36.028611],  # Aracati, Brazil
    "BRB": [-1.576389, -48.502778],  # Barreirinhas, Brazil
    "HAM": [53.630389, 9.988228],  # Hamburg, Germany
    "RBR": [-9.866667, -67.383333],  # Rio Branco, Brazil
    "RVY": [-30.886667, -55.529444],  # Rivera, Uruguay
    "SOD": [-23.478611, -47.490833],  # Sorocaba, Brazil
}
