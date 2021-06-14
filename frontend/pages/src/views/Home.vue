<template>
    <div class="home">
        <div class="background">
        </div>
        <div class="content" id="home">
            <div class="logo">
                <img alt="logo" style='border-radius:50%; width:300px; height:300px;' src="./../assets/logo.png"
                     id="logo">
            </div>
            <div class="text unSelect">
                <span style="display: block;font-size: 2em;margin-bottom: 0.67em;font-weight: bold;">Welcome to <span
                        class="thin_font">AnalyticMesh</span></span>
            </div>
            <div class="button">
                <Button value="Enter" :clicked="enter_click" color='#42b983' border_color="#4fc08d"></Button>
            </div>
        </div>
    </div>
</template>

<script>
    // @ is an alias to /src
    import Button from '@/components/Button.vue'

    export default {
        name: 'Home',
        components: {
            Button
        },
        methods: {
            enter_click: function () {
                this.$router.push('operate');
            }
        }
    }
</script>
<style>
    .home {
        display: flex;
        width: 100%;
        min-height: 100vh;
    }

    .background {
        position: fixed;
        width: 100%;
        height: 100vh;
        background: url('./../assets/background.png');
        background-size: 100% 100%;
        opacity: 0.2;
    }

    .thin_font {
        font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif;
        font-weight: 300;
    }

    #home {
        flex: 1;
        height: 500px;
        margin: auto;
        z-index: 2;
    }

    #logo:hover {
        animation: rotation ease-in-out infinite;
        animation-duration: 3s;
    }

    @keyframes rotation {
        from {
            transform: rotate(-720deg);
        }
    }
</style>