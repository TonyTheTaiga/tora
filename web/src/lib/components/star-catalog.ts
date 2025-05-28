export interface RealStarData {
  id: number;
  name: string;
  ra: number; // Right Ascension in decimal degrees
  dec: number; // Declination in decimal degrees
  mag: number; // Visual Magnitude
  spectralType: string;
}

export const BRIGHT_STARS_CATALOG: RealStarData[] = [
  {
    id: 1,
    name: "Sirius",
    ra: 101.287,
    dec: -16.716,
    mag: -1.46,
    spectralType: "A1V",
  },
  {
    id: 3,
    name: "Arcturus",
    ra: 212.083,
    dec: 19.182,
    mag: -0.05,
    spectralType: "K1.5III",
  },
  {
    id: 4,
    name: "Vega",
    ra: 279.235,
    dec: 38.783,
    mag: 0.03,
    spectralType: "A0V",
  }, // Visible in the early evening, setting in the WNW.
  {
    id: 5,
    name: "Capella",
    ra: 79.172,
    dec: 45.998,
    mag: 0.08,
    spectralType: "G8III",
  },
  {
    id: 6,
    name: "Rigel",
    ra: 77.587,
    dec: -8.201,
    mag: 0.13,
    spectralType: "B8Ia",
  },
  {
    id: 7,
    name: "Procyon",
    ra: 114.823,
    dec: 5.225,
    mag: 0.34,
    spectralType: "F5IV-V",
  },
  {
    id: 9,
    name: "Betelgeuse",
    ra: 88.792,
    dec: 7.407,
    mag: 0.5,
    spectralType: "M1-2Ia-Iab",
  },
  {
    id: 11,
    name: "Aldebaran",
    ra: 67.876,
    dec: 16.509,
    mag: 0.85,
    spectralType: "K5III",
  },
  {
    id: 12,
    name: "Antares",
    ra: 247.351,
    dec: -26.431,
    mag: 1.06,
    spectralType: "M1.5Iab-b",
  }, // Visible in the pre-dawn sky, rising in the SE.
  {
    id: 13,
    name: "Spica",
    ra: 204.423,
    dec: -11.161,
    mag: 1.04,
    spectralType: "B1V",
  }, // Rises after midnight.
  {
    id: 14,
    name: "Pollux",
    ra: 116.326,
    dec: 28.026,
    mag: 1.15,
    spectralType: "K0IIIb",
  },
  {
    id: 15,
    name: "Fomalhaut",
    ra: 344.407,
    dec: -29.627,
    mag: 1.16,
    spectralType: "A3V",
  }, // Visible in the evening sky in the south.
  {
    id: 17,
    name: "Regulus",
    ra: 152.091,
    dec: 11.967,
    mag: 1.35,
    spectralType: "B7V",
  }, // Rises mid-evening.
  {
    id: 18,
    name: "Adhara",
    ra: 105.109,
    dec: -28.98,
    mag: 1.5,
    spectralType: "B2Iab",
  },
  {
    id: 19,
    name: "Castor",
    ra: 110.37,
    dec: 31.888,
    mag: 1.58,
    spectralType: "A1V",
  },
  {
    id: 21,
    name: "Shaula",
    ra: 260.672,
    dec: -37.1,
    mag: 1.63,
    spectralType: "B1.5V",
  }, // Visible in the pre-dawn sky, rising in the SE.
  {
    id: 22,
    name: "Bellatrix",
    ra: 81.336,
    dec: 6.35,
    mag: 1.64,
    spectralType: "B2III",
  },
  {
    id: 23,
    name: "Elnath",
    ra: 77.014,
    dec: 28.629,
    mag: 1.65,
    spectralType: "B7III",
  },
  {
    id: 25,
    name: "Alnilam",
    ra: 84.664,
    dec: -1.2,
    mag: 1.69,
    spectralType: "B0Ia",
  },
  {
    id: 26,
    name: "Alnair",
    ra: 326.687,
    dec: -46.863,
    mag: 1.7,
    spectralType: "B7V",
  }, // Visible in the early evening in the S/SSW.
  {
    id: 27,
    name: "Alioth",
    ra: 195.42,
    dec: 55.959,
    mag: 1.76,
    spectralType: "A1IIIp",
  }, // Circumpolar from NYC.
  {
    id: 28,
    name: "Dubhe",
    ra: 165.75,
    dec: 61.699,
    mag: 1.79,
    spectralType: "K0III",
  }, // Circumpolar from NYC.
  {
    id: 29,
    name: "Mirfak",
    ra: 49.33,
    dec: 49.866,
    mag: 1.8,
    spectralType: "F5Ib",
  }, // Circumpolar (or very nearly) from NYC.
  {
    id: 30,
    name: "Wezen",
    ra: 107.037,
    dec: -26.059,
    mag: 1.83,
    spectralType: "F8Ia",
  },
  {
    id: 31,
    name: "Kaus Australis",
    ra: 276.992,
    dec: -34.331,
    mag: 1.85,
    spectralType: "B9.5III",
  }, // Visible in the early evening, setting in the WSW.
  {
    id: 32,
    name: "Alkaid",
    ra: 206.94,
    dec: 49.317,
    mag: 1.86,
    spectralType: "B3V",
  }, // High northern declination, effectively circumpolar from NYC.
  {
    id: 33,
    name: "Sargas",
    ra: 244.305,
    dec: -45.305,
    mag: 1.87,
    spectralType: "F1II",
  }, // Visible in the pre-dawn sky, rising in the SE.
  {
    id: 35,
    name: "Kochab",
    ra: 228.618,
    dec: 74.004,
    mag: 2.07,
    spectralType: "K4III",
  }, // Circumpolar from NYC.
  {
    id: 36,
    name: "Polaris",
    ra: 37.954,
    dec: 89.264,
    mag: 1.98,
    spectralType: "F7Ib-II",
  }, // Circumpolar from NYC (the North Star).
];
