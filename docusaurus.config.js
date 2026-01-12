// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Educational Robotics Book',
  tagline: 'A comprehensive guide to robotics and AI in educational environments',
  favicon: 'img/favicon.svg',

  // Set the production url of your site here
  url: 'https://jamshaid-aziz121.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub Pages deployment, it's typically '/<project-name>/'
  baseUrl: '/hackathon_001/',

  // GitHub pages deployment config.
  organizationName: 'Jamshaid-aziz121', // Usually your GitHub org/user name.
  projectName: 'hackathon_001', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  trailingSlash: undefined,

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['urdu'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Jamshaid-aziz121/hackathon_001/',
        },
        blog: false, // Disable blog functionality
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/logo.png',
      navbar: {
        title: 'Educational Robotics Book',
        logo: {
          alt: 'Robotics Book Logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Read',
          },
          {
            href: 'https://github.com/Jamshaid-aziz121/hackathon_001',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro/getting-started',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Jamshaid-aziz121/hackathon_001',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Educational Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};


export default config;
