import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#effcf6',
          100: '#d8f7e8',
          200: '#b4eed2',
          300: '#83dfb3',
          400: '#4dc88f',
          500: '#28ad75',
          600: '#1a8b5f',
          700: '#176f4d',
          800: '#16593f',
          900: '#134a35'
        }
      },
      fontFamily: {
        sans: ['Manrope', 'ui-sans-serif', 'system-ui']
      }
    }
  },
  plugins: [],
} satisfies Config
