window.onload = () => {
    const interval = setInterval(() => {
        const root = document.getElementsByTagName('gradio-app')?.[0]?.shadowRoot;
        if (root) {
            clearInterval(interval);
            
            const cards = root.querySelectorAll('.card');
            
            let christianMangaJPG;

            for (let card of cards) {
                const bgPNG = card.style.backgroundImage;

                if (!bgPNG &&card.getAttribute('onclick').includes('NSFW')) {
                    console.log('NSFW',card.getAttribute('onclick'), card.getAttribute('onclick').includes('NSFW'))
                    
                    card.style.backgroundImage = christianMangaJPG;
                }

                if (bgPNG && bgPNG !== '' && bgPNG.includes('extensions')) {
                    const bgJPG = bgPNG.replace('.png', '.jpg');

                    if (!christianMangaJPG) {
                        const bgArr = bgJPG.split('/');
                        const i = bgArr.indexOf('extensions');
                        const endMtime = '&mtime' + bgJPG.split('&mtime')[1];
                        christianMangaJPG = bgArr.slice(0, i + 2).join('/') + '/embeddings/Christian Manga for NSFW content.jpg' + endMtime;
                    }

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