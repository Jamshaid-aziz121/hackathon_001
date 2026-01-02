// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro/getting-started',
    {
      type: 'category',
      label: 'Module 1 - The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/core-concepts',
        'module-1-ros2/tooling-sdks',
        'module-1-ros2/implementation-walkthrough',
        'module-1-ros2/case-study',
        'module-1-ros2/mini-project',
        'module-1-ros2/debugging'
      ],
    },
    {
      type: 'category',
      label: 'Module 2 - The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-simulation/core-concepts',
        'module-2-simulation/tooling-sdks',
        'module-2-simulation/implementation-walkthrough',
        'module-2-simulation/case-study',
        'module-2-simulation/mini-project',
        'module-2-simulation/debugging'
      ],
    },
    {
      type: 'category',
      label: 'Module 3 - The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-ai-perception/core-concepts',
        'module-3-ai-perception/tooling-sdks',
        'module-3-ai-perception/implementation-walkthrough',
        'module-3-ai-perception/case-study',
        'module-3-ai-perception/mini-project',
        'module-3-ai-perception/debugging'
      ],
    },
    {
      type: 'category',
      label: 'Module 4 - Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/core-concepts',
        'module-4-vla/tooling-sdks',
        'module-4-vla/implementation-walkthrough',
        'module-4-vla/case-study',
        'module-4-vla/mini-project',
        'module-4-vla/debugging'
      ],
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/end-to-end-projects',
        'tutorials/advanced-integration',
        'tutorials/cross-module-reference'
      ],
    },
    'references/citations',
  ],
};

export default sidebars;