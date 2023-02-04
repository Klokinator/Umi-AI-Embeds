window.onload = () => {
    const interval = setInterval(() => {
        const root = document.getElementsByTagName('gradio-app')?.[0]?.shadowRoot;
        if (root) {
            clearInterval(interval);
            
            const cards = root.querySelectorAll('.card');
            for (let card of cards) {
                const bgPNG = card.style.backgroundImage;
                

                if (bgPNG && bgPNG !== '' && bgPNG.includes('extensions')) {
                    const bgJPG = bgPNG.replace('.png', '.jpg');
                    fetch(bgJPG.replace('url("', '').replace('")', '')).then(res => {
                        if (!res.ok) return;
                        card.style.backgroundImage = bgJPG;
                        card.addEventListener('mouseover', () => {
                            card.style.backgroundImage = bgPNG;
                        });
                        card.addEventListener('mouseout', () => {
                            card.style.backgroundImage = bgJPG;
                        });
                    }).catch(() => {});
                }
            }
        }
    }, 100);
}                