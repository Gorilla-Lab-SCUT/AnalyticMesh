<template>
    <div class="operate">
        <div class="nav">
            <img alt="logo" style='border-radius:50%; width:40px; height:40px;padding:5px'
                 src="./../assets/logo.png" id="logo">
            <div class="nav_title">
                AnalyticMesh
            </div>
        </div>
        <div class="content">
            <div class="left scroller2">
                <div id="import_section" class="section" @mouseenter="import_mouse_enter"
                     @mouseleave="import_mouse_leave">
                    <div v-show="import_show_filename" style="color:#4fc08d;font-size:1.5rem">
                        {{import_filename}}
                    </div>
                    <div v-show="!import_show_filename">
                        <Button value="Import" color='#4fc08d' border_color="#4fc08d" font_size="1rem"
                                width="140px" :is_file=true :clicked="import_btn_event" file_accept=".onnx">
                        </Button>
                    </div>
                </div>
                <div v-if="status> 0">
                    <div v-if='status > 0' id="preview_section" class="section"
                         style="text-align: left; margin: 0 10px">
                        <span style="color:#4fc08d">Model Preview：</span>
                        <span class="unSelect" style="font-size:0.4rem">Click to view in full screen</span>
                        <div class="box scroller"
                             style="height:calc((100vh - 280px) / 7 * 4);border-color:#4fc08d;background:#4fc08d05"
                             ref="preview_modal">
                            <div @click="open_preview_screen">
                                <img style="width:100%;height:100%;" ref="preview_img">
                            </div>
                        </div>
                    </div>
                    <div v-if='status > 0' id="config_section" class="section"
                         style="text-align: left; margin: 0 10px !important">
                        <span style="color:#4f7bc0">Configure：</span>
                        <div class="box scroller"
                             style="height:calc((100vh - 280px) / 7 * 2 + 40px);border-color:#4f7bc0;display:flex;flex-direction: column;background:#4f7bc005">
                            <v_Input text="ISO" style="width:100%" font_size="1rem" active_color="#4f7bc0"
                                     place_holder="please input float number" v-model="ISO"
                                     rule="[0-9]*(\.[0-9]*)?" align="center"
                                     info='The isolevel of the isosurface you want to extract'></v_Input>
                            <TwoLayerEmbedding text="TRIGGER"
                                               :content=config_trigger_content
                                               style="margin-top:10px;flex:1"
                                               font_size="1rem" active_color="#4f7bc0"
                                               modal_background_color="#ffffff"
                                               v-model="TRIGGER"
                                               info="The triggering method to find those initial points on the isosurface"
                            ></TwoLayerEmbedding>
                            <Button value="Submit" color='#4f7bc0' border_color="#4f7bc0" font_size="0.8rem"
                                    width="50px" :clicked="submit_btn_event"
                                    style="text-align: center;margin-bottom:3px;margin-top:5px"></Button>
                        </div>
                    </div>
                </div>
                <div v-if="status > 1">
                    <div v-if='status > 1' id="log_section" class="section"
                         style="text-align: left; margin: 0 10px !important;">
                        <span style="color:#c04f59">Simplify：</span>
                        <div class="box scroller3"
                             style="height:calc((100vh - 280px) / 7);border-color:#c04f59;overflow-y: auto;text-align:center;background:#c04f5905;display:flex;flex-direction: column;">
                            <div style="display:flex;flex:1;margin-top:6px">
                                <span style="color:#c04f59;padding-right:10px;width: 50px;">{{simplify_value}}</span>
                                <input type="range" id="dur" value="0.50" min="0.001" max="1.000" step="0.001"
                                       v-model="simplify_value" ref="simplify_slider">
                            </div>
                            <Button value="Simplify" color='#c04f59' border_color="#c04f59" font_size="0.8rem"
                                    width="50px" :clicked="simplify_bn_event"
                                    style="margin-top:5px;margin-bottom:3px"></Button>
                        </div>
                    </div>
                    <div v-if='status > 1' id="export_section" class="section">
                        <Button value="Export" color='#c04f59' border_color="#c04f59" font_size="1rem"
                                width="140px" :clicked="export_bn_event"></Button>
                    </div>
                </div>
            </div>
            <div class="right">
                <div id="three_container"></div>
                <div id="right_readme">
                    <div class="right_readme_content" ref="readme_body">
                        <iframe src="./readme.html" width="100%" height="100%" frameborder="0"
                                scrolling="auto"></iframe>
                    </div>
                    <div class="right_readme_btn unSelect" @click="open_readme" v-if="!readme_open">
                        user guide
                    </div>
                    <div class="right_readme_btn unSelect" v-else @click="close_readme">
                        close
                    </div>
                </div>
            </div>
        </div>
        <div ref="loading" class="loading">
            <div v-if="status === 0" style="font-size:2rem;line-height:100vh;text-align: center">
                loading……
            </div>
            <div style="width:50vw;margin:auto;margin-top:100px;height:calc(100vh - 150px);overflow-y: auto;"
                 v-else-if="status >= 1" class="scroller4">
                <div style="width:calc(100% - 12px);margin:auto">
                    <ProcessBar :value="running_model_processbar" :total="running_model_total" title="Running Model"
                                :text="running_model_text" width="100%" height="10px"
                                v-show="status === 1"></ProcessBar>
                    <ProcessBar :value="download_model_processbar" :total="download_model_total"
                                title="Downloading Mesh"
                                :text="download_model_text" width="100%" height="10px"
                                style="margin-top:50px"></ProcessBar>
                    <Terminal style="width:100%;height:300px;margin:auto;margin-top:50px" ref="terminal"></Terminal>
                    <Button value="View the mesh" color="white" border_color="white" background="rgba(0,0,0,0)"
                            style="margin-top:50px" :clicked="exit_submit" v-if="submit_finish"></Button>
                </div>
            </div>
        </div>
        <div class="preview_all_screen scroller" ref="preview_all_screen" @click="close_preview_screen">
            <div style="width:50vw;margin:auto;">

            </div>
        </div>
    </div>
</template>

<script>
    import Button from '@/components/Button.vue'
    import v_Input from '@/components/Input.vue'
    import TwoLayerEmbedding from '@/components/TwoLayerEmbedding.vue'
    import ProcessBar from '@/components/ProcessBar.vue'
    import Terminal from '@/components/Terminal.vue'

    import * as Three from 'three'
    import {PLYLoader} from 'three/examples/jsm/loaders/PLYLoader.js'
    import {WS_ADDRESS} from './../assets/index.js'


    let OrbitControls = require('three-orbit-controls')(Three);
    let Base64 = require('js-base64').Base64;
    let FileSaver = require('file-saver');


    export function readFile(file) {
        return new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = (e) => {
                resolve(e.target.result);
            };
            reader.onerror = reject;
        });
    }

    export default {
        name: "Operate",
        components: {
            Button,
            v_Input,
            TwoLayerEmbedding,
            ProcessBar,
            Terminal
        },
        data() {
            return {
                camera: null,
                scene: null,
                renderer: null,
                mesh: null,
                status: 0,
                clock: null,
                controls: null,

                import_filename: '',
                import_show_filename: false,

                ws: null,
                ws_callback: {
                    'import': this.import_callback,
                    'submit': this.submit_callback,
                    'simplify': this.simplify_callback
                },

                ISO: '0.0',
                TRIGGER: {
                    method: '',
                    args: {}
                },

                config_trigger_content: {
                    gradient_descent: {
                        init_num: {
                            default: 1024,
                            relu: "[0-9]*",
                            info: 'How many points you want to find'
                        },
                        lr_max: {
                            default: 1e-2,
                            relu: "[0-9]*(\\.[0-9]*)?",
                            info: 'The maximum initial value of the learning rate'
                        },
                        avg_eps: {
                            default: 1e-3,
                            relu: "[0-9]*(\\.[0-9]*)?",
                            info: 'The maximum of error'
                        },
                    },
                    sphere_tracing: {
                        init_num: {
                            default: 1024,
                            relu: "[0-9]*",
                            info: 'How many points you want to find'
                        },
                        step_size_max: {
                            default: 1.0,
                            relu: "[0-9]*(\\.[0-9]*)?",
                            info: 'The initial step size to approach the surface'
                        },
                        avg_eps: {
                            default: 1e-3,
                            relu: "[0-9]*(\\.[0-9]*)?",
                            info: 'The maximum of error'
                        },
                    },
                    dichotomy: {
                        init_num: {
                            default: 1024,
                            relu: "[0-9]*",
                            info: 'How many points you want to find'
                        },
                        iter_max: {
                            default: 100,
                            relu: "[0-9]*",
                            info: 'The maximum number of iterations allowed'
                        },
                        avg_eps: {
                            default: 1e-3,
                            relu: "[0-9]*(\\.[0-9]*)?",
                            info: 'The maximum of error'
                        },
                    }
                },

                running_model_processbar: 0,
                running_model_total: 100,
                running_model_text: '0/100',
                running_model_interval: null,

                download_model_processbar: 0,
                download_model_total: 100,
                download_model_text: '0/100',

                ply_model: null,
                temp_chunks: [],
                submit_finish: false,

                simplify_value: 0.5,

                readme_open: false
            }
        },
        methods: {
            init: function () {
                let container = document.getElementById('three_container');
                this.camera = new Three.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1500);
                this.camera.lookAt(new Three.Vector3(0, 0, 0));

                this.camera.position.z = 100;
                this.scene = new Three.Scene();

                let light1 = new Three.DirectionalLight(0xffffff);
                light1.position.set(1500, 1500, 1500);
                let light4 = new Three.DirectionalLight(0xffffff);
                light4.position.set(1500, -1500, -1500);
                let light6 = new Three.DirectionalLight(0xffffff);
                light6.position.set(-1500, -1500, 1500);
                let light8 = new Three.DirectionalLight(0xffffff);
                light8.position.set(-1500, 1500, -1500);

                this.scene.add(light1);
                this.scene.add(light4);
                this.scene.add(light6);
                this.scene.add(light8);


                this.renderer = new Three.WebGLRenderer({antialias: true});
                this.renderer.setClearColor(0xb8e7d1);
                this.renderer.setSize(container.clientWidth, container.clientHeight);
                container.appendChild(this.renderer.domElement);
                this.renderer.render(this.scene, this.camera);

                this.clock = new Three.Clock();

                this.controls = new OrbitControls(this.camera, container);
                this.controls.target = new Three.Vector3(0, 0, 0);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.25;
                this.controls.enableZoom = true;
                this.controls.autoRotate = false;
                this.controls.enablePan = true;
            },

            init_scene: function () {
                if (this.mesh !== null) {
                    this.scene.remove(this.mesh);
                    this.mesh = null;
                }
                if (this.ply_model !== null) {
                    this.ply_model = null;
                }
            },

            init_config: function () {
                this.ISO = '0';
                this.TRIGGER = {
                    method: '',
                    args: {}
                };
            },

            init_processbar: function () {
                this.running_model_processbar = 0;
                this.download_model_processbar = 0;
                if (this.running_model_interval !== null) {
                    window.clearInterval(this.running_model_interval);
                    this.running_model_interval = null;
                }
                this.running_model_text = '0/100';
                this.download_model_text = '0/100';
                this.running_model_total = 100;
                this.download_model_total = 100;
                this.submit_finish = false;
            },

            animate: function () {
                let delta = this.clock.getDelta();
                this.controls.update(delta);
                requestAnimationFrame(this.animate);
                this.renderer.render(this.scene, this.camera);
            },

            import_mouse_enter: function () {
                this.import_show_filename = false;
            },
            import_mouse_leave: function () {
                if (this.import_filename !== '') {
                    this.import_show_filename = true;
                }
            },

            import_btn_event: function (event) {
                if (event.target.files.length === 0) {
                    return;
                }
                this.import_filename = '';
                this.import_show_filename = false;
                this.init_config();
                this.init_scene();

                this.status = 0;
                if (this.ws !== null) {
                    this.ws.close();
                }
                this.ws = new WebSocket(WS_ADDRESS);
                this.ws.onmessage = this.ws_message;
                this.ws.onerror = this.ws_error;
                this.$refs.loading.setAttribute('style', 'display:block');
                this.ws.onopen = () => {
                    this.import_filename = event.target.files[0].name;
                    this.import_show_filename = true;
                    readFile(event.target.files[0]).then(e => {
                        this.ws.send(JSON.stringify({
                            action: 'import',
                            data: e
                        }));
                        // let upload the same file triggering the onchange event;
                        event.target.value = null;
                    });
                };
            },
            ws_message: function (e) {
                let json_data;
                try {
                    json_data = JSON.parse(e.data);
                } catch (e) {
                    console.log('parse error:', e.data);
                    return;
                }
                if (Object.keys(json_data).indexOf('callback') === -1 || Object.keys(json_data).indexOf('data') === -1) {
                    console.log('error data:', json_data);
                    return;
                }
                return this.ws_callback[json_data['callback']](json_data['data']);
            },
            ws_error: function () {
                this.status = 0;
                this.$refs.loading.setAttribute('style', 'display:none');
                this.import_filename = '';
                this.import_show_filename = false;

                alert('the connection is closed, please refresh and try again');
            },
            import_callback: function (data) {
                this.$refs.loading.setAttribute('style', 'display:none');
                this.status = 1;
                this.$nextTick(() => {
                    this.$refs.preview_img.setAttribute('src', 'data:image/png;base64,' + data);
                });
            },
            submit_callback: function (data) {
                if (data.success === 0) {
                    this.$refs.loading.setAttribute('style', 'display:none');
                    this.$nextTick(() => {
                        alert('server error:' + data.msg);
                    });
                    return;
                }
                if (data.type === 'estimate time') {
                    this.$refs.terminal.addLine('server: Running model estimate time: ' + data.long + ' s');
                    this.running_model_total = data.long;
                    this.running_model_processbar = 0;
                    this.running_model_interval = setInterval(() => {
                        this.running_model_processbar += 0.1;
                        this.running_model_text = (this.running_model_processbar + '').substring(0, 4) + '/' + (this.running_model_total + '').substring(0, 4);
                    }, 100);
                } else if (data.type === 'running') {
                    this.$refs.terminal.addLine('server: Running model...');
                } else if (data.type === 'result') {
                    const _data = data.data;
                    for (let i in _data) {
                        this.$refs.terminal.addLine('server: ' + i + ' : ' + _data[i] + ' s');
                    }
                } else if (data.type === 'download model size') {
                    if (this.running_model_interval !== null) {
                        window.clearInterval(this.running_model_interval);
                        this.running_model_interval = null;
                        this.running_model_total = this.running_model_processbar;
                        this.running_model_text = (this.running_model_processbar + '').substring(0, 4) + '/' + (this.running_model_total + '').substring(0, 4);
                    }
                    this.temp_chunks = [];
                    this.ply_model = new Uint8Array(data.real_size);
                    this.$refs.terminal.addLine('server: Download mesh size: ' + data.real_size + ' Bytes');
                    this.download_model_total = data.long;
                    this.download_model_processbar = 0;
                } else if (data.type === 'downloading') {
                    this.download_model_processbar += 1;
                    const chunk_data = Base64.toUint8Array(data.data);
                    this.temp_chunks.push(chunk_data);
                    this.download_model_text = (this.download_model_processbar + '').substring(0, 4) + '/' + (this.download_model_total + '').substring(0, 4)
                } else if (data.type === 'finish') {
                    this.$nextTick(() => {
                        this.$refs.terminal.addLine('server: Finish downloading');
                        this.merge_chunks();
                        this.updateScene();
                        this.$refs.terminal.addLine('web: Render successfully');
                        this.$refs.terminal.addLine('web: Please click the button below to view the mesh');
                        this.submit_finish = true;
                    });
                }
            },

            simplify_callback: function (data) {
                if (data.success === 0) {
                    this.$refs.loading.setAttribute('style', 'display:none');
                    this.$nextTick(() => {
                        alert('server error:' + data.msg);
                    });
                    return;
                }
                if (data.type === 'running') {
                    this.$refs.terminal.addLine('server: Simplifying model...');
                } else if (data.type === 'download model size') {
                    this.temp_chunks = [];
                    this.ply_model = new Uint8Array(data.real_size);
                    this.$refs.terminal.addLine('server: Download mesh size: ' + data.real_size + ' Bytes');
                    this.download_model_total = data.long;
                    this.download_model_processbar = 0;
                } else if (data.type === 'downloading') {
                    this.download_model_processbar += 1;
                    const chunk_data = Base64.toUint8Array(data.data);
                    this.temp_chunks.push(chunk_data);
                    this.download_model_text = (this.download_model_processbar + '').substring(0, 4) + '/' + (this.download_model_total + '').substring(0, 4)
                } else if (data.type === 'finish') {
                    this.$nextTick(() => {
                        this.$refs.terminal.addLine('server: Finish downloading');
                        this.merge_chunks();
                        this.updateScene();
                        this.$refs.terminal.addLine('web: Render successfully');
                        this.$refs.terminal.addLine('web: Please click the button below to view the mesh');
                        this.submit_finish = true;
                    });
                }
            },

            merge_chunks: function () {
                this.$refs.terminal.addLine('web: Start merging');
                let c = 0;
                for (let i in this.temp_chunks) {
                    const temp = this.temp_chunks[i];
                    for (let j in temp) {
                        this.ply_model[c] = temp[j];
                        c++;
                    }
                }
                this.temp_chunks = [];
                this.$refs.terminal.addLine('web: Finish merging');
            },
            updateScene: function () {
                let loader = new PLYLoader();
                const geometry = loader.parse(this.ply_model.buffer);
                let m = new Three.MeshStandardMaterial({
                    color: 0xcccccc,
                    flatShading: true,
                });
                m.side = Three.DoubleSide;
                this.mesh = new Three.Mesh(geometry, m);
                this.mesh.scale.set(10, 10, 10);
                this.scene.add(this.mesh);
            },

            submit_btn_event: function () {
                if (this.ws === null) {
                    alert('please upload modal firstly');
                    this.status = 0;
                    return;
                }
                if (this.ISO === '' || !this.check_data(this.TRIGGER.args) || this.TRIGGER.method === '') {
                    alert('data is empty or data is error');
                    return;
                }
                this.status = 1;
                this.init_processbar();
                this.init_scene();
                this.$refs.terminal.clear();
                this.$refs.loading.setAttribute('style', 'display:block');
                const data = {
                    iso: this.ISO,
                    triggering: this.TRIGGER,
                };
                this.ws.send(JSON.stringify({
                    action: 'submit',
                    data: data
                }));
            },
            exit_submit: function () {
                this.status = 2;
                this.$refs.loading.setAttribute('style', 'display:none');
                this.close_readme();
            },
            check_data: function (dict) {
                for (let index in dict) {
                    if (dict[index] === '' || parseFloat(dict[index]) != dict[index]) {
                        return false;
                    }
                }
                return true;
            },
            open_preview_screen: function () {
                this.$refs.preview_all_screen.children[0].innerHTML = this.$refs.preview_modal.innerHTML;
                this.$refs.preview_all_screen.style.display = 'block';
            },
            close_preview_screen: function () {
                this.$refs.preview_all_screen.style.display = 'none';
            },
            simplify_bn_event: function () {
                if (this.ws === null) {
                    alert('please upload modal firstly');
                    this.status = 0;
                    return;
                }
                this.status = 2;
                this.init_processbar();
                this.init_scene();
                this.$refs.terminal.clear();
                this.$refs.loading.setAttribute('style', 'display:block');
                this.ws.send(JSON.stringify({
                    action: 'simplify',
                    data: this.simplify_value
                }));
            },
            export_bn_event: function () {
                let blob = new Blob([this.ply_model.buffer]);
                FileSaver.saveAs(blob, this.import_filename.replace('.onnx', '.ply'));
            },
            open_readme: function () {
                this.readme_open = true;
                this.$refs.readme_body.style = 'height: calc(100vh - 101px)';
            },
            close_readme: function () {
                this.readme_open = false;
                this.$refs.readme_body.style = 'height: 0';

            }
        },
        mounted() {
            this.init();
            this.animate();
            this.open_readme();
        }
    }
</script>

<style scoped>
    .operate {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100vh;
        min-height: 500px;
        min-width: 500px;
    }

    .nav {
        height: 50px;
        background: #65b967;
        width: 100%;
        justify-content: center;
        text-align: center;
        display: flex;
        flex-direction: row;
        box-shadow: 0 0 10px 0 #8f908f;
        z-index: 100;
    }

    .nav_title {
        line-height: 50px;
        font-size: 1.5rem;
        margin-left: 5px;
        font-weight: 700;
        color: white
    }

    .content {
        flex: 1;
        display: flex;
        flex-direction: row;
        min-height: 450px;
    }

    .left {
        width: 300px;
        border-right: 1px solid gray;
        box-shadow: 0 0 5px 0 gray;
        overflow-y: auto;
        padding-bottom: 5px;
        z-index: 50;
        overflow-x: hidden;
    }

    .right {
        flex: 1;
    }

    .section {
        margin-top: 5px !important;
        animation: show .5s forwards;
        transform: translateY(100px);
        transition: 0.5s;
    }

    @keyframes show {
        to {
            visibility: visible;
            transform: translateY(0)
        }
    }

    .box {
        border: 1px solid darkseagreen;
        border-radius: 6px;
        overflow-y: auto;
        padding: 6px;
    }

    .scroller::-webkit-scrollbar {
        /*滚动条整体样式*/
        width: 6px; /*高宽分别对应横竖滚动条的尺寸*/
        height: 6px;
    }

    .scroller::-webkit-scrollbar-thumb {
        /*滚动条里面小方块*/
        border-radius: 3px;
        box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.2);
        background: #4fc08d;
    }

    .scroller::-webkit-scrollbar-track {
        /*滚动条里面轨道*/
        box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
        background: #ededed;
        border-radius: 3px;
    }

    .scroller2::-webkit-scrollbar {
        /*滚动条整体样式*/
        width: 6px; /*高宽分别对应横竖滚动条的尺寸*/
        height: 6px;
    }

    .scroller2::-webkit-scrollbar-thumb {
        /*滚动条里面小方块*/
        border-radius: 3px;
        box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.2);
        background: darkseagreen;
    }

    .scroller2::-webkit-scrollbar-track {
        /*滚动条里面轨道*/
        box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
        background: #ededed;
        border-radius: 3px;
    }

    .scroller3::-webkit-scrollbar {
        /*滚动条整体样式*/
        width: 6px; /*高宽分别对应横竖滚动条的尺寸*/
        height: 6px;
    }

    .scroller3::-webkit-scrollbar-thumb {
        /*滚动条里面小方块*/
        border-radius: 3px;
        box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.2);
        background: #c04f59;
    }

    .scroller3::-webkit-scrollbar-track {
        /*滚动条里面轨道*/
        box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
        background: #ededed;
        border-radius: 3px;
    }

    .scroller4::-webkit-scrollbar {
        /*滚动条整体样式*/
        width: 6px; /*高宽分别对应横竖滚动条的尺寸*/
        height: 6px;
    }

    .scroller4::-webkit-scrollbar-thumb {
        /*滚动条里面小方块*/
        border-radius: 3px;
        box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.2);
        background: black;
    }

    .scroller4::-webkit-scrollbar-track {
        /*滚动条里面轨道*/
        box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
        background: #ededed;
        border-radius: 3px;
    }

    #three_container {
        width: 100%;
        height: calc(100% - 1px);
    }

    #right_readme {
        position: relative;
        top: calc(51px - 100vh);
        transition: 1s;
    }

    .right_readme_btn {
        text-align: center;
        width: 120px;
        background: darkolivegreen;
        margin: auto;
        height: 30px;
        border-radius: 10px;
        line-height: 30px;
        font-size: 1.3rem;
        color: white;
        box-shadow: 0 0 10px 0 #8f908f;
        cursor: pointer;
        opacity: 0.5;
    }

    .right_readme_btn:hover {
        box-shadow: 0 0 10px 0 black;
        opacity: 1;
    }

    .right_readme_btn:active {
        box-shadow: 0 0 0 0;
    }

    .right_readme_content {
        background: white;
        height: 0;
        transition: 1s;
        box-shadow: 0 0 10px 0 #8f908f;
        width: calc(100% - 100px);
        margin: auto;
        border-radius: 3px;
        overflow-y: hidden;
    }

    .loading {
        position: fixed;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.6);
        z-index: 1000;
        display: none;
        text-align: center;
        color: white
    }

    .preview_all_screen {
        position: fixed;
        width: calc(100vw - 12px);
        height: calc(100vh - 12px);
        background: rgba(0, 0, 0, 0.6);
        z-index: 1000;
        display: none;
        text-align: center;
        overflow-y: auto;
        border: 1px solid #4fc08d;
        border-radius: 10px;
        padding: 5px;
    }

    input[type="range"] {
        width: 100%;
        -webkit-appearance: none;
        height: 8px;
        border-radius: 4px;
        outline: none;
        background: linear-gradient(to right, white 0%, #b94273) no-repeat;
        border: 1px solid #bd6086;
    }

    /* -webkit-slider-thumb仅对谷歌浏览器有效 */
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        background-color: #b98597;
        width: 8px;
        height: 20px;
        border-radius: 4px;
        cursor: pointer;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
        background: #b94273;
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