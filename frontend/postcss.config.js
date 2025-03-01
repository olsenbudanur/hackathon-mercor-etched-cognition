// postcss.config.js
module.exports = {
    plugins: [
      require('tailwindcss'),    // This will import Tailwind
      require('autoprefixer')    // This will automatically add vendor prefixes
    ]
  }