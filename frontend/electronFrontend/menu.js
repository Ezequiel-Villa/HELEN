const {app, Menu} = require('electron')

const setMenu = () => {
    const template = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Exit',
                    click: () => {
                        app.quit()
                    }
                }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                {
                    label: 'Copy',
                    role: 'copy'
                },
                {
                    label: 'Paste',
                    role: 'paste'
                }
            ]
        }
    ]

}

module.exports = {
    setMenu
}