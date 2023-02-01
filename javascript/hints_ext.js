
// hints no longer work iirc
globalThis.titles = {
    ...globalThis.titles,
    "Verbose logging": "Displays UmiAI log messages. Useful when prompt crafting, or debugging file-path errors.",
    "Cache files": "Cache .txt and .yaml files at runtime. Speeds up prompt generation. Disable if you're editing wildcard files to see changes instantly.",
    "Same prompt in batch": 'Same prompt will be used for all generated images in a batch.',
    '**negative keywords**': 'Collect and add **negative keywords** from wildcards to Negative Prompts. Needed for fine-tuned UMI Preset Prompts.',
    'Static wildcards': 'Always picks the same random/wildcard options when using a static seed.'
};

// window.onload = () => {
//     const interval = setInterval(() => {
//         const root = document.getElementsByTagName('gradio-app')?.[0]?.shadowRoot;
//         if (root) {
//             clearInterval(interval);
            
//             const cards = root.querySelectorAll('.card');
//             for (let card of cards) {
//                 const bgPNG = card.style.backgroundImage;
//                 const bgJPG = bgPNG.replace('.png', '.jpg');



//                 console.log('card', card, bgPNG, bgJPG)
//                 if (bgPNG && bgPNG !== '' && bgPNG.includes('extensions')) {
//                     card.style.backgroundImage = bgJPG;
//                     card.addEventListener('mouseover', () => {
//                         card.style.backgroundImage = bgPNG;
//                     });
//                     card.addEventListener('mouseout', () => {
//                         card.style.backgroundImage = bgJPG;
//                     });                            
//                 }
//             }
//         }
//     }, 100);
// }
